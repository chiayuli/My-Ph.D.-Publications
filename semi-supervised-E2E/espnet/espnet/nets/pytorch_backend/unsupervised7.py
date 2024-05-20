import logging

import argparse
import numpy as np
import torch
import math
from chainer import reporter
from itertools import groupby
import editdistance
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.initialization import lecun_normal_init_parameters
from espnet.nets.pytorch_backend.initialization import set_forget_bias_to_one
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.decoders import decoder_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for
from espnet.utils.fill_missing_args import fill_missing_args
from espnet.nets.pytorch_backend.rnn.argument import (
    add_arguments_rnn_encoder_common,  # noqa: H301
    add_arguments_rnn_decoder_common,  # noqa: H301
    add_arguments_rnn_attention_common,  # noqa: H301
)
import chainer
import six
import os

import espnet.nets.pytorch_backend.e2e_asr as base

CTC_LOSS_THRESHOLD = 10000

def mmd(xs,ys,beta=1.0):
    Nx = xs.shape[0]
    Ny = ys.shape[0]
    Kxy = torch.matmul(xs,ys.t())
    dia1 = torch.sum(xs*xs,1)
    dia2 = torch.sum(ys*ys,1)
    Kxy = Kxy-0.5*dia1.unsqueeze(1).expand(Nx,Ny)
    Kxy = Kxy-0.5*dia2.expand(Nx,Ny)
    Kxy = torch.exp(beta*Kxy).sum()/Nx/Ny

    Kx = torch.matmul(xs,xs.t())
    Kx = Kx-0.5*dia1.unsqueeze(1).expand(Nx,Nx)
    Kx = Kx-0.5*dia1.expand(Nx,Nx)
    Kx = torch.exp(beta*Kx).sum()/Nx/Nx

    Ky = torch.matmul(ys,ys.t())
    Ky = Ky-0.5*dia2.unsqueeze(1).expand(Ny,Ny)
    Ky = Ky-0.5*dia2.expand(Ny,Ny)
    Ky = torch.exp(beta*Ky).sum()/Ny/Ny

    return Kx+Ky-2*Kxy


class _Det(torch.autograd.Function):
    """
    Matrix determinant. Input should be a square matrix
    """

    @staticmethod
    def forward(ctx, x):
        #output = x.potrf().diag().prod()**2
        output = x.cholesky().diag().prod()**2
        output = x.new([output])
        ctx.save_for_backward(x, output)
        # ctx.save_for_backward(u, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, output = ctx.saved_variables
        # u, output = ctx.saved_variables
        grad_input = None

        if ctx.needs_input_grad[0]:
            # TODO TEST
            grad_input = grad_output * output * x.inverse().t()
            # grad_input = grad_output * output * torch.potrf(u).t()

        return grad_input

def det(x):
    # u = torch.potrf(x)
    return _Det.apply(x)


class LogDet(torch.autograd.Function):
    """
    Matrix log determinant. Input should be a square matrix
    """

    @staticmethod
    def forward(ctx, x, eps=0.0):
        output = torch.log(x.cholesky().diag() + eps).sum() * 2
        output = x.new([output])
        ctx.save_for_backward(x, output)
        # ctx.save_for_backward(u, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, output = ctx.saved_variables
        # u, output = ctx.saved_variables
        grad_input = None

        if ctx.needs_input_grad[0]:
            # TODO TEST
            grad_input = grad_output * x.inverse().t()
            # grad_input = grad_output * torch.potrf(u).t()
        return grad_input

def logdet(x):
    # u = torch.potrf(x)
    return LogDet.apply(x)


def test_det():
    x = Variable(torch.rand(3, 3) / 10.0 + torch.eye(3).float(), requires_grad=True)
    torch.autograd.gradcheck(det, (x,), eps=1e-4, atol=0.1, rtol=0.1)

def test_logdet():
    x = Variable(torch.rand(3, 3) + torch.eye(3).float() * 3 , requires_grad=True)
    d = det(x).log()
    d.backward()
    gd = x.grad.clone()
    ld = logdet(x)
    x.grad = None
    ld.backward()
    gld = x.grad
    np.testing.assert_allclose(d.data.numpy(), ld.data.numpy())
    np.testing.assert_allclose(gd.data.numpy(), gld.data.numpy())

def cov(xs, m=None):
    assert xs.dim() == 2
    if m is None:
        m = xs.mean(0, keepdim=True)
    assert m.size() == (1, xs.size(1))
    return (xs - m).t().mm(xs - m) / xs.size(0)

threshold = torch.nn.functional.threshold

def unclamp_(x, eps):
    """
    >>> a = torch.FloatTensor([0.0, 1.0, -0.1, 0.1])
    >>> unclamp(a, 0.5)
    [0.5, 1.0, -0.5, 0.5]
    """
    ng = x.abs() < eps
    sign = x.sign()
    fill_value = sign.float() * eps + (sign == 0).float() * eps
    return x.masked_fill_(ng, 0) + ng.float() * fill_value

def gauss_kld(xs, ys, use_logdet=False, eps=float(np.finfo(np.float32).eps)):
    n_batch, n_hidden = xs.size()
    xm = xs.mean(0, keepdim=True)
    ym = ys.mean(0, keepdim=True)
    xcov = cov(xs, xm)
    ycov = cov(ys, ym)
    xcov += torch.diag(xcov.diag() + eps)
    ycov += torch.diag(ycov.diag() + eps)
    if use_logdet:
        log_ratio = logdet(ycov) - logdet(xcov)
    else:
        log_ratio = torch.log(threshold(det(ycov), eps, eps)) - torch.log(threshold(det(xcov), eps, eps))
    ycovi = ycov.inverse()
    xym = xm - ym  # (1, n_hidden)
    hess = xym.mm(ycovi).mm(xym.t())
    tr = torch.trace(ycovi.mm(xcov))
    return 0.5 * (log_ratio + tr + hess - n_hidden).squeeze()


class EmbedRNN(torch.nn.Module):
    def __init__(self, n_in, n_out, n_layers=1):
        super(EmbedRNN, self).__init__()
        self.embed = torch.nn.Embedding(n_in, n_out, padding_idx=0)
        self.rnn = torch.nn.LSTM(n_out, n_out, n_layers,
                                 bidirectional=True, batch_first=True)
        self.merge = torch.nn.Linear(n_out * 2, n_out)

    def forward(self, xpad, xlen):
        """
        :param xpad: (batchsize x max(xlen)) LongTensor
        :return hpad: (batchsize x max(xlen) x n_out) FloatTensor
        :return hlen: length list of int. hlen == xlen
        """
        h = self.embed(xpad)
        hpack = pack_padded_sequence(h, xlen, batch_first=True)
        hpack, states = self.rnn(hpack)
        hpad, hlen = pad_packed_sequence(hpack, batch_first=True)
        b, t, o = hpad.shape
        hpad = self.merge(hpad.contiguous().view(b * t, o)).view(b, t, -1)
        return hpad, hlen


class MMSEDecoder(torch.nn.Module):
    """
    hidden-to-speech decoder with a MMSE criterion

    TODO(karita): use Tacotron-like structure
    """
    def __init__(self, eprojs, odim, dlayers, dunits, att, verbose=0):
        super(MMSEDecoder, self).__init__()
        self.dunits = dunits
        self.dlayers = dlayers
        self.in_linear = torch.nn.Linear(odim, dunits)
        self.decoder = torch.nn.ModuleList()
        self.decoder += [torch.nn.LSTMCell(dunits + eprojs, dunits)]
        for l in six.moves.range(1, self.dlayers):
            self.decoder += [torch.nn.LSTMCell(dunits, dunits)]
        self.output = torch.nn.Linear(dunits, odim)

        self.loss = None
        self.att = att
        self.dunits = dunits
        self.verbose = verbose

    def zero_state(self, hpad):
        return Variable(hpad.data.new(hpad.size(0), self.dunits).zero_())

    def forward(self, hpad, hlen, ypad, ylen):
        '''Decoder forward

        :param hs:
        :param ys:
        :return:
        '''
        hpad = base.mask_by_length(hpad, hlen, 0)
        self.loss = None

        # get dim, length info
        batch = ypad.size(0)
        olength = ypad.size(1)

        # initialization
        c_list = [self.zero_state(hpad)]
        z_list = [self.zero_state(hpad)]
        for l in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hpad))
            z_list.append(self.zero_state(hpad))
        att_w = None
        z_all = []
        self.att.reset()  # reset pre-computation of h
        att_weight_all = []  # for debugging

        # pre-computation of embedding
        eys = self.in_linear(ypad.view(batch * olength, -1)).view(batch, olength, -1)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hpad, hlen, z_list[0], att_w)
            ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list[0], c_list[0] = self.decoder[0](ey, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            z_all.append(z_list[-1])
            att_weight_all.append(att_w.data)  # for debugging

        z_all = torch.stack(z_all, dim=1).view(batch * olength, self.dunits)
        # compute loss
        y_all = self.output(z_all).view(batch, olength, -1)
        ym = base.mask_by_length(y_all, ylen)
        tm = base.mask_by_length(ypad, ylen)
        self.loss = torch.sum((ym - tm) ** 2)
        self.loss *= (np.mean(ylen))
        logging.info('att loss:' + str(self.loss.data))
        return self.loss, att_weight_all


class Discriminator(torch.nn.Module):
    def __init__(self, idim, odim):
        super(Discriminator, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.ReLU(),
            torch.nn.Linear(odim, odim),
            torch.nn.ReLU(),
            torch.nn.Linear(odim, 1)
        )

    def forward(self, spack, tpack):
        ns = spack.size(0)
        nt = tpack.size(0)
        input = torch.cat((spack, tpack), dim=0)
        predict = self.seq(input)
        target = input.data.new(ns + nt, 1)
        target[:ns] = 0
        target[ns:] = 1
        target = Variable(target)
        return -torch.nn.functional.binary_cross_entropy_with_logits(predict, target)

class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss_ctc, loss_att, acc, cer_ctc, cer, wer, mtl_loss):
        """Report at every step."""
        reporter.report({"loss_ctc": loss_ctc}, self)
        reporter.report({"loss_att": loss_att}, self)
        reporter.report({"acc": acc}, self)
        reporter.report({"cer_ctc": cer_ctc}, self)
        reporter.report({"cer": cer}, self)
        reporter.report({"wer": wer}, self)
        logging.info("mtl loss:" + str(mtl_loss))
        reporter.report({"loss": mtl_loss}, self)

class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2E.encoder_add_arguments(parser)
        E2E.attention_add_arguments(parser)
        E2E.decoder_add_arguments(parser)
        return parser

    @staticmethod
    def encoder_add_arguments(parser):
        """Add arguments for the encoder."""
        group = parser.add_argument_group("E2E encoder setting")
        group = add_arguments_rnn_encoder_common(group)
        return parser

    @staticmethod
    def attention_add_arguments(parser):
        """Add arguments for the attention."""
        group = parser.add_argument_group("E2E attention setting")
        group = add_arguments_rnn_attention_common(group)
        return parser

    @staticmethod
    def decoder_add_arguments(parser):
        """Add arguments for the decoder."""
        group = parser.add_argument_group("E2E decoder setting")
        group = add_arguments_rnn_decoder_common(group)
        return parser

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        if isinstance(self.enc, torch.nn.ModuleList):
            return self.enc[0].conv_subsampling_factor * int(np.prod(self.subsample))
        else:
            return self.enc.conv_subsampling_factor * int(np.prod(self.subsample))

    def __init__(self, idim, odim, args):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)

        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)
        self.speech_text_ratio = args.speech_text_ratio
        self.cyc_loss = args.cyc_loss
        self.idt_loss = args.idt_loss
        self.idt_enc_t = args.idt_enc_t
        self.idt_enc_s = args.idt_enc_s
        self.mtlalpha = args.mtlalpha
        assert 0.0 <= self.mtlalpha <= 1.0, "mtlalpha should be [0.0, 1.0]"
        self.etype = args.etype
        self.verbose = args.verbose
        # NOTE: for self.build method
        args.char_list = getattr(args, "char_list", None)
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.space = args.sym_space
        self.blank = args.sym_blank
        self.reporter = Reporter()

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1
       # subsample info
        self.subsample = get_subsample(args, mode="asr", arch="rnn")

        # label smoothing info
        if args.lsm_type and os.path.isfile(args.train_json):
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(
                odim, args.lsm_type, transcript=args.train_json
            )
        else:
            labeldist = None

        if getattr(args, "use_frontend", False):  # use getattr to keep compatibility
            self.frontend = frontend_for(args, idim)
            self.feature_transform = feature_transform_for(args, (idim - 1) * 2)
            idim = args.n_mels
        else:
            self.frontend = None

        if hasattr(args, "unsupervised_loss"):
            self.unsupervised_loss = args.unsupervised_loss
        else:
            self.unsupervised_loss = None
        if hasattr(args, "use_batchnorm") and args.use_batchnorm:
            self.batchnorm = torch.nn.BatchNorm1d(args.eprojs)
        else:
            self.batchnorm = None

        # encoder
        self.enc_t = EmbedRNN(odim, args.eprojs)
        self.enc = base.encoder_for(args, idim, self.subsample)
        self.enc_rnn0 = getattr(self.enc.enc[0], "birnn%d" % (0))
        self.enc_rnn1 = getattr(self.enc.enc[0], "birnn%d" % (1))
        self.enc_rnn2 = getattr(self.enc.enc[0], "birnn%d" % (2))
        self.enc_rnn3 = getattr(self.enc.enc[0], "birnn%d" % (3))
        self.enc_rnn4 = getattr(self.enc.enc[0], "birnn%d" % (4))
        self.enc_common_rnn = getattr(self.enc.enc[0], "birnn%d" % (args.elayers-1))
        self.enc_merge0 = getattr(self.enc.enc[0], "bt%d" % (0))
        self.enc_merge1 = getattr(self.enc.enc[0], "bt%d" % (1))
        self.enc_merge2 = getattr(self.enc.enc[0], "bt%d" % (2))
        self.enc_merge3 = getattr(self.enc.enc[0], "bt%d" % (3))
        self.enc_merge4 = getattr(self.enc.enc[0], "bt%d" % (4))
        self.enc_common_merge = getattr(self.enc.enc[0], "bt%d" % (args.elayers-1))

        # ctc
        self.ctc = base.ctc_for(args, odim)

        # attention
        self.att = base.att_for(args)

        # decoder
        self.dec = base.decoder_for(args, odim, self.sos, self.eos, self.att, labeldist)

        # weight initialization
        self.init_like_chainer()

        # options for beam search
        if args.report_cer or args.report_wer:
            recog_args = {
                "beam_size": args.beam_size,
                "penalty": args.penalty,
                "ctc_weight": args.ctc_weight,
                "maxlenratio": args.maxlenratio,
                "minlenratio": args.minlenratio,
                "lm_weight": args.lm_weight,
                "rnnlm": args.rnnlm,
                "nbest": args.nbest,
                "space": args.sym_space,
                "blank": args.sym_blank,
            }

            self.recog_args = argparse.Namespace(**recog_args)
            self.report_cer = args.report_cer
            self.report_wer = args.report_wer
        else:
            self.report_cer = False
            self.report_wer = False
        self.rnnlm = None

        self.logzero = -10000000000.0
        self.loss = None
        self.acc = None

    def init_like_chainer(self):
        """Initialize weight like chainer.

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)
        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        self.enc_t.embed.weight.data.normal_(0, 1)
        self.dec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for i in six.moves.range(len(self.dec.decoder)):
          set_forget_bias_to_one(self.dec.decoder[i].bias_ih)

    def sort_variables(self, xs, sorted_index):
        xs = [xs[i] for i in sorted_index]
        xs = [base.to_cuda(self, Variable(torch.from_numpy(xx))) for xx in xs]
        xlens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
        return xs, xlens

    def forward_common(self, xpad, xlen):
        # hpad, hlen = self.enc_common_rnn(xpad, xlen)
        xpack = pack_padded_sequence(xpad, xlen, batch_first=True)
        hpack, states = self.enc_common_rnn(xpack)
        hpad, hlen = pad_packed_sequence(hpack, batch_first=True)
        b, t, o = hpad.shape
        hpad = torch.tanh(self.enc_common_merge(hpad.contiguous().view(b * t, o)).view(b, t, -1))
        return hpad, hlen

    def forward_uncommon(self, xpad, xlen):
        # hpad, hlen = self.enc_common_rnn(xpad, xlen)
        xpack0 = pack_padded_sequence(xpad, xlen, batch_first=True)
        # layer-0
        hpack0, states = self.enc_rnn0(xpack0)
        hpad0, hlen0 = pad_packed_sequence(hpack0, batch_first=True)
        b, t, o = hpad0.shape
        hpad0 = torch.tanh(self.enc_merge0(hpad0.contiguous().view(b * t, o)).view(b, t, -1))
        hpack0 = pack_padded_sequence(hpad0, hlen0, batch_first=True)
        # layer-1
        hpack1, states = self.enc_rnn1(hpack0)
        hpad1, hlen1 = pad_packed_sequence(hpack1, batch_first=True)
        b, t, o = hpad1.shape
        hpad1 = torch.tanh(self.enc_merge1(hpad1.contiguous().view(b * t, o)).view(b, t, -1))
        hpack1 = pack_padded_sequence(hpad1, hlen1, batch_first=True)
        # layer-2
        hpack2, states = self.enc_rnn2(hpack1)
        hpad2, hlen2 = pad_packed_sequence(hpack2, batch_first=True)
        b, t, o = hpad2.shape
        hpad2 = torch.tanh(self.enc_merge2(hpad2.contiguous().view(b * t, o)).view(b, t, -1))
        hpack2 = pack_padded_sequence(hpad2, hlen2, batch_first=True)
        # layer-3
        hpack3, states = self.enc_rnn3(hpack2)
        hpad3, hlen3 = pad_packed_sequence(hpack3, batch_first=True)
        b, t, o = hpad3.shape
        hpad3 = torch.tanh(self.enc_merge3(hpad3.contiguous().view(b * t, o)).view(b, t, -1))
        hpack3 = pack_padded_sequence(hpad3, hlen3, batch_first=True)
        # layer-4
        hpack4, states = self.enc_rnn4(hpack3)
        hpad4, hlen4 = pad_packed_sequence(hpack4, batch_first=True)
        b, t, o = hpad4.shape
        hpad4 = torch.tanh(self.enc_merge4(hpad4.contiguous().view(b * t, o)).view(b, t, -1))
        return hpad4, hlen4

    def forward(self, xs_pad, ilens, ys_pad, olens, supervised=False, discriminator=None, only_encoder=False):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """

        # 0. Frontend
        if self.frontend is not None:
            hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
            hs_pad, hlens = self.feature_transform(hs_pad, hlens)
        else:
            hs_pad, hlens = xs_pad, ilens

        if supervised or not self.training:
            # 1. Encoder
            hs_pad, hlens, _ = self.enc(hs_pad, hlens)

            # 2. CTC loss
            if self.mtlalpha == 0:
                self.loss_ctc = None
            else:
                self.loss_ctc = self.ctc(hs_pad, hlens, ys_pad)

            # 3. attention loss
            if self.mtlalpha == 1:
                self.loss_att, acc = None, None
            else:
                self.loss_att, acc, _, _ = self.dec(hs_pad, hlens, ys_pad)
            self.acc = acc

            # 4. compute cer without beam search
            if self.mtlalpha == 0 or self.char_list is None:
                cer_ctc = None
            else:
                cers = []

                y_hats = self.ctc.argmax(hs_pad).data
                for i, y in enumerate(y_hats):
                    y_hat = [x[0] for x in groupby(y)]
                    y_true = ys_pad[i]

                    seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                    seq_true = [
                        self.char_list[int(idx)] for idx in y_true if int(idx) != -1
                    ]
                    seq_hat_text = "".join(seq_hat).replace(self.space, " ")
                    seq_hat_text = seq_hat_text.replace(self.blank, "")
                    seq_true_text = "".join(seq_true).replace(self.space, " ")

                    hyp_chars = seq_hat_text.replace(" ", "")
                    ref_chars = seq_true_text.replace(" ", "")
                    if len(ref_chars) > 0:
                        cers.append(
                            editdistance.eval(hyp_chars, ref_chars) / len(ref_chars)
                        )

                cer_ctc = sum(cers) / len(cers) if cers else None

            # 5. compute cer/wer
            if self.training or not (self.report_cer or self.report_wer):
                cer, wer = 0.0, 0.0
                # oracle_cer, oracle_wer = 0.0, 0.0
            else:
                if self.recog_args.ctc_weight > 0.0:
                    lpz = self.ctc.log_softmax(hs_pad).data
                else:
                    lpz = None

                word_eds, word_ref_lens, char_eds, char_ref_lens = [], [], [], []
                nbest_hyps = self.dec.recognize_beam_batch(
                    hs_pad,
                    torch.tensor(hlens),
                    lpz,
                    self.recog_args,
                    self.char_list,
                    self.rnnlm,
                )
                # remove <sos> and <eos>
                y_hats = [nbest_hyp[0]["yseq"][1:-1] for nbest_hyp in nbest_hyps]
                for i, y_hat in enumerate(y_hats):
                    y_true = ys_pad[i]

                    seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                    seq_true = [
                        self.char_list[int(idx)] for idx in y_true if int(idx) != -1
                    ]
                    seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, " ")
                    seq_hat_text = seq_hat_text.replace(self.recog_args.blank, "")
                    seq_true_text = "".join(seq_true).replace(self.recog_args.space, " ")

                    hyp_words = seq_hat_text.split()
                    ref_words = seq_true_text.split()
                    word_eds.append(editdistance.eval(hyp_words, ref_words))
                    word_ref_lens.append(len(ref_words))
                    hyp_chars = seq_hat_text.replace(" ", "")
                    ref_chars = seq_true_text.replace(" ", "")
                    char_eds.append(editdistance.eval(hyp_chars, ref_chars))
                    char_ref_lens.append(len(ref_chars))

                wer = (
                    0.0
                    if not self.report_wer
                    else float(sum(word_eds)) / sum(word_ref_lens)
                )
                cer = (
                    0.0
                    if not self.report_cer
                    else float(sum(char_eds)) / sum(char_ref_lens)
                )

            alpha = self.mtlalpha

            if alpha == 0:
                self.loss = self.loss_att
                loss_att_data = float(self.loss_att)
                loss_ctc_data = None
            elif alpha == 1:
                self.loss = self.loss_ctc
                loss_att_data = None
                loss_ctc_data = float(self.loss_ctc)
            else:
                self.loss = alpha * self.loss_ctc + (1 - alpha) * self.loss_att
                loss_att_data = float(self.loss_att)
                loss_ctc_data = float(self.loss_ctc)

            loss_data = float(self.loss)
            if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
                self.reporter.report(
                    loss_ctc_data, loss_att_data, acc, cer_ctc, cer, wer, loss_data
                )
            else:
                logging.warning("loss (=%f) is not correct", loss_data)
            return self.loss
        else:
            #for name, layer in self.enc_common_rnn.named_modules():
                #if isinstance(layer, torch.nn.ReLU):
             #   logging.warning("[enc_common_rnn] name: %s  layer:%s"%(name, layer))

            #for name, layer in self.enc.named_modules():
                #if isinstance(layer, torch.nn.ReLU):
            #    logging.warning("[enc] name: %s  layer:%s"%(name, layer))
            # forward encoder for text
            hypad, hylens = self.enc_t(ys_pad, olens)

            # forward common encoder
            hypad, hylens = self.forward_common(hypad, hylens)
            hypack = pack_padded_sequence(hypad, hylens, batch_first=True)

            if self.unsupervised_loss is not None and self.unsupervised_loss != "None":
                hxpad, hxlens, _ = self.enc(xs_pad, ilens)
                hxpack = pack_padded_sequence(hxpad, hxlens, batch_first=True)

                if hxpack is None:
                    logging.warning("AHAKDJFAKDFJALKDSJ NONE!")

                if self.batchnorm:
                    hxpack = PackedSequence(self.batchnorm(hxpack.data), hxpack.batch_sizes)
                    hypack = PackedSequence(self.batchnorm(hypack.data), hypack.batch_sizes)

                if only_encoder:
                    return hxpack, hypack
                if self.unsupervised_loss == "variance":
                    loss_unsupervised = torch.cat((hxpack.data, hypack.data), dim=0).var(1).mean()
                if self.unsupervised_loss == "gauss":
                    loss_unsupervised = gauss_kld(hxpack.data, hypack.data)
                if self.unsupervised_loss == "gausslogdet":
                    loss_unsupervised = gauss_kld(hxpack.data, hypack.data, use_logdet=True)
                if self.unsupervised_loss == "mmd":
                    loss_unsupervised = mmd(hxpack.data, hypack.data)
                if self.unsupervised_loss == "gan":
                    loss_unsupervised = discriminator(hxpack.data, hypack.data)
            else:
                loss_unsupervised = 0.0
                if only_encoder:
                    hxpad, hxlens = self.enc(xs_pad, ilens)
                    hxpack = pack_padded_sequence(hxpad, hxlens, batch_first=True)
                    return hxpack, hypack

            # 3. forward decoders
            loss_text, acc, att_t, ys_hat = self.dec(hypad, hylens, ys_pad)
            #logging.warning("loss_hidden %f loss_text %f acc %f, ys_hat %s" %(loss_unsupervised, loss_text, acc, ys_hat))
            #loss = self.speech_text_ratio * loss_unsupervised + (1.0 - self.speech_text_ratio) * loss_text

            # 4. high-level representation cyclic loss
            loss_cyc_h = 0
            #if (self.cyc_loss is not None) and (self.cyc_loss != 0):
            if True:
                ### high-level representation from text (eq 10)
                ys_hat = [to_device(self, to_torch_tensor(yy).long()) for yy in ys_hat]
                ys_hat_pad = pad_list(ys_hat, 0)
                hypad2, hylens2 = self.enc_t(ys_hat_pad, hylens)
                hypad2, hylen2 =self.forward_common(hypad2,hylens2)
                loss_cyc_h_from_y = (hypad - hypad2)**2
                ### high-level representation from speech (eq 10)
                hs_pad, hlens, _ = self.enc(xs_pad, ilens)
                _, _, _, ys_hat = self.dec(hs_pad, hlens, ys_pad)
                ys_hat = [to_device(self, to_torch_tensor(yy).long()) for yy in ys_hat]
                ys_hat_pad = pad_list(ys_hat, 0)
                hypad3, hylens3 = self.enc_t(ys_hat_pad, hylens)
                hypad3, hylen3 =self.forward_common(hypad3,hylens3)
                hxpack = pack_padded_sequence(hs_pad, hlens, batch_first=True)
                hypack = pack_padded_sequence(hypad3, hylens3, batch_first=True)
                loss_cyc_h_from_x = mmd(hxpack.data, hypack.data)
                #loss_cyc_h = loss_cyc_h_from_y.mean() + loss_cyc_h_from_x
                ### high-level representation from speech (ep 11)
                hs_pad, hlens, _ = self.enc(xs_pad, ilens)
                _, _, _, ys_hat = self.dec(hs_pad, hlens, ys_pad)
                ys_hat = [to_device(self, to_torch_tensor(yy).long()) for yy in ys_hat]
                ys_hat_pad = pad_list(ys_hat, 0)
                hypad3, hylens3 = self.enc_t(ys_hat_pad, hylens)
                hypack3 = pack_padded_sequence(hypad3, hylens3, batch_first=True)
                hxpad4, hxlens4 = self.forward_uncommon(xs_pad, ilens)
                hxpack4 = pack_padded_sequence(hxpad4, hxlens4, batch_first=True)
                loss_e_cyc_i_from_x = mmd(hxpack4.data, hypack3.data)
                #logging.warning("eq 11. loss_e_cyc_i_from_x %f"%(loss_e_cyc_i_from_x))

            # 5. identity loss
            loss_idt = 0
            if True :
                ### identity loss for Encoder (h -> Enc -> h)
                hypad4, hylens4 = self.enc_t(ys_pad, olens)
                hypad5, hylens5 = self.forward_common(hypad4, hylens4) # Enc(Emb(t)): high-level rep. h
                hypad6, hylens6 = self.forward_common(hypad5, hylens5)
                loss_idt_enc_t = (hypad6 - hypad5)**2
                #logging.warning("loss_idt_enc_t.mean() %f",loss_idt_enc_t.mean())
                hxpad7, hxlens7, _ = self.enc(xs_pad, ilens) # Enc(s): h
                hypad8, hylens8 = self.forward_common(hxpad7, hxlens7)
                loss_idt_enc_s = (hypad8 - hxpad7)**2
                #logging.warning("loss_idt_enc_s.mean() %f",loss_idt_enc_s.mean())
                ### identity loss for Decoder (t -> Dec -> t)
                hypadt, hylenst = self.enc_t(ys_pad, olens)
                loss_idt_dec, acc, att_t, ys_hat = self.dec(hypadt, hylenst, ys_pad)
                #logging.warning("loss_idt_dec.mean() %f acc %f"%(loss_idt_dec.mean(), acc))
                loss_idt_enc = (loss_idt_enc_t.mean() * self.idt_enc_t) + (loss_idt_enc_s.mean() * self.idt_enc_s)
                loss_idt = (loss_idt_enc + loss_idt_dec)/2

            loss = (self.speech_text_ratio * (loss_cyc_h_from_x+loss_e_cyc_i_from_x)) +\
                   ((1.0 - self.speech_text_ratio) * (loss_text+loss_cyc_h_from_y.mean()))
            #loss = (self.speech_text_ratio * loss_unsupervised) +\
            #       ((1.0 - self.speech_text_ratio) * loss_text) + \
            #       (self.cyc_loss * loss_cyc_h) + \
            #       (self.idt_loss * loss_idt) 
            #logging.warning("loss %f loss_text %f loss_speech %f "%(loss, (loss_text+loss_cyc_h_from_y.mean()+loss_idt_enc_t.mean()), (loss_cyc_h_from_x+loss_idt_enc_s.mean())))
            return loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.dec, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: input acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        ilens = [x.shape[0]]

        # subsample frame
        x = x[:: self.subsample[0], :]
        p = next(self.parameters())
        h = torch.as_tensor(x, device=p.device, dtype=p.dtype)
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)

        # 0. Frontend
        if self.frontend is not None:
            enhanced, hlens, mask = self.frontend(hs, ilens)
            hs, hlens = self.feature_transform(enhanced, hlens)
        else:
            hs, hlens = hs, ilens

        # 1. encoder
        hs, _, _ = self.enc(hs, hlens)
        return hs.squeeze(0)

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        """E2E beam search.

        :param ndarray x: input acoustic feature (T, D)
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        hs = self.encode(x).unsqueeze(0)
        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(hs)[0]
        else:
            lpz = None

        # 2. Decoder
        # decode the first utterance
        y = self.dec.recognize_beam(hs[0], lpz, recog_args, char_list, rnnlm)
        return y

    def recognize_batch(self, xs, recog_args, char_list, rnnlm=None):
        """E2E batch beam search.

        :param list xs: list of input acoustic feature arrays [(T_1, D), (T_2, D), ...]
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[:: self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)

        # 0. Frontend
        if self.frontend is not None:
            enhanced, hlens, mask = self.frontend(xs_pad, ilens)
            hs_pad, hlens = self.feature_transform(enhanced, hlens)
        else:
            hs_pad, hlens = xs_pad, ilens

        # 1. Encoder
        hs_pad, hlens, _ = self.enc(hs_pad, hlens)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(hs_pad)
            normalize_score = False
        else:
            lpz = None
            normalize_score = True

        # 2. Decoder
        hlens = torch.tensor(list(map(int, hlens)))  # make sure hlens is tensor
        y = self.dec.recognize_beam_batch(
            hs_pad,
            hlens,
            lpz,
            recog_args,
            char_list,
            rnnlm,
            normalize_score=normalize_score,
        )

        if prev:
            self.train()
        return y

    def enhance(self, xs):
        """Forward only in the frontend stage.

        :param ndarray xs: input acoustic feature (T, C, F)
        :return: enhaned feature
        :rtype: torch.Tensor
        """
        if self.frontend is None:
            raise RuntimeError("Frontend does't exist")
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[:: self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)
        enhanced, hlensm, mask = self.frontend(xs_pad, ilens)
        if prev:
            self.train()
        return enhanced.cpu().numpy(), mask.cpu().numpy(), ilens

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, olens):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            # 0. Frontend
            if self.frontend is not None:
                hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
                hs_pad, hlens = self.feature_transform(hs_pad, hlens)
            else:
                hs_pad, hlens = xs_pad, ilens

            # 1. Encoder
            hpad, hlens, _ = self.enc(hs_pad, hlens)

            # 2. Decoder
            att_ws = self.dec.calculate_all_attentions(hpad, hlens, ys_pad)
        self.train()
        return att_ws
    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad, olens):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        probs = None
        if self.mtlalpha == 0:
            return probs

        self.eval()
        with torch.no_grad():
            # 0. Frontend
            if self.frontend is not None:
                hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
                hs_pad, hlens = self.feature_transform(hs_pad, hlens)
            else:
                hs_pad, hlens = xs_pad, ilens

            # 1. Encoder
            hpad, hlens, _ = self.enc(hs_pad, hlens)

            # 2. CTC probs
            probs = self.ctc.softmax(hpad).cpu().numpy()
        self.train()
        return probs

    def subsample_frames(self, x):
        """Subsample speeh frames in the encoder."""
        # subsample frame
        x = x[:: self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_device(self, torch.from_numpy(np.array(x, dtype=np.float32)))
        h.contiguous()
        return h, ilen

