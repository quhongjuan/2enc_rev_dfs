# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, input11: torch.Tensor, input22: torch.Tensor, input33: torch.Tensor=None) -> torch.Tensor:
        if len(input11.shape) > 2:
            input1 = torch.squeeze(input11, dim=0)
            input2 = torch.squeeze(input22, dim=0)
            # input3 = torch.squeeze(input33)
        else:
            input1 = input11
            input2 = input22
            #input3 = input33


        assert len(input1.shape) == 2
        assert input1.shape[0] == input2.shape[0]
        #assert input1.shape[0] == input3.shape[0]
        batch_size = input1.shape[0]

        sim12 = torch.cosine_similarity(input1, input2, dim=1)  # [batch_size]
        #sim13 = torch.cosine_similarity(input1, input3, dim=1)
        sim_positive = torch.exp(sim12) # + torch.exp(sim13)      # [batch_size]

        index = [i for i in range(batch_size)]
        index1 = index
        index = index1[1:]
        index.append(index1[0])
        input111 = input1[index1]
        cos_sim = torch.cosine_similarity(input1, input111)
        sim_negative = torch.exp(cos_sim)  # + torch.exp(torch.cosine_similarity(input2, input2[index1])) #+ torch.exp(torch.cosine_similarity(input3, input3[index1]))

        for i in range(1, batch_size - 1):
            index1 = index
            index = index1[1:]
            index.append(index1[0])
            input111 = input1[index1]
            cos_sim2 = torch.cosine_similarity(input1, input111)
            sim_negative = sim_negative + torch.exp(cos_sim2) #+ torch.exp(torch.cosine_similarity(input2, input2[index1])) + torch.exp(torch.cosine_similarity(input3, input3[index1]))
        loss = torch.log(sim_positive / (sim_positive + sim_negative)) * -1
        return torch.mean(loss)

def create_absolute_positional_embedding(batch_size=3, seq_len=200, em_dim=512):

    positional_embedding = torch.zeros([seq_len, em_dim], dtype=torch.float)
    pos = torch.arange(seq_len)[:, None].float()
    omega = torch.arange(em_dim//2, dtype=torch.float)
    omega /= em_dim//2
    omega = 1./(10000**omega)
    omega = omega[None, :]
    out = pos @ omega
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    positional_embedding[:, 0::2] = emb_sin
    positional_embedding[:, 1::2] = emb_cos
    positional_embedding = positional_embedding.repeat([batch_size, 1, 1])
    return positional_embedding

def pe(one_batch):
    # add positional embedding for one_batch
    batch_size, seq_len, em_dim = one_batch.size()[0], one_batch.size()[1], one_batch.size()[2]
    positional_embedding = torch.zeros([seq_len, em_dim], dtype=torch.float).to(one_batch.device)
    pos = torch.arange(seq_len)[:, None].float().to(one_batch.device)
    omega = torch.arange(em_dim // 2, dtype=torch.float).to(one_batch.device)
    omega /= em_dim // 2
    omega = 1. / (10000 ** omega)
    omega = omega[None, :]
    out = pos @ omega
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    positional_embedding[:, 0::2] = emb_sin
    positional_embedding[:, 1::2] = emb_cos
    positional_embedding = positional_embedding.repeat([batch_size, 1, 1])
    return positional_embedding + one_batch

class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """

    def __init__(self, encoder, decoder1, config, args, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        # self.embedding = nn.Embedding(embedding_dim=args.hidden_size, num_embeddings=args.src_vocab_size, padding_idx=1) #trans  
        # self.embedding_nl = self.embedding  # nn.Embedding(embedding_dim=args.hidden_size, num_embeddings=args.tgt_vocab_size, padding_idx=1) #trans  
        self.encoder = encoder
        self.decoder1 = decoder1
        # self.decoder2 = decoder2
        # self.decoder3 = decoder3
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #self.cat_lin = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.args = args
        self.constrativeLoss = ContrastiveLoss()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id


    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head, self.encoder.embeddings.word_embeddings)
    def forward(self, source_ids=None, source_mask=None, dfs_ids=None, dfs_mask=None, rev_ids=None, rev_mask=None,
                target_ids=None, target_mask=None, args=None, src_real_len=None, pdg_real_len=None):

        #src_outputs = self.encoder(pe(self.embedding(source_ids)).permute([1, 0, 2]).contiguous(), src_key_padding_mask=(source_mask - 1).bool())               #[200,2,512]
        src_outputs = self.encoder(source_ids, attention_mask=source_mask)

        #rev_outputs = self.encoder(self.embedding(rev_ids).permute([1, 0, 2]).contiguous(),src_key_padding_mask=(source_mask - 1).bool())
        """src_outputs = self.encoder(source_ids, src_key_padding_mask=source_mask)
        dfs_outputs = self.encoder(dfs_ids, src_key_padding_mask=dfs_mask)
        rev_outputs = self.encoder(rev_ids, src_key_padding_mask=rev_mask)"""

        """src_encoder_output = src_outputs.permute([1, 0, 2]).contiguous()
        dfs_encoder_output = dfs_outputs.permute([1, 0, 2]).contiguous()# [2,200,512]
        rev_encoder_output = rev_outputs.permute([1, 0, 2]).contiguous()"""

        src_encoder_output = src_outputs[0].permute([1, 0, 2]).contiguous()

        #dfs_encoder_output = dfs_outputs.contiguous()  # [200,2,512]
        #rev_encoder_output = rev_outputs.contiguous()
        """src_encoder_output = src_outputs[0].permute([1, 0, 2]).contiguous()
        dfs_encoder_output = dfs_outputs[0].permute([1, 0, 2]).contiguous()
        rev_encoder_output = rev_outputs[0].permute([1, 0, 2]).contiguous()"""

        #encoder_output = self.cat_lin(torch.cat((src_encoder_output, dfs_encoder_output, rev_encoder_output), 2))
        if target_ids is not None:
            #dfs_outputs = self.encoder(pe(self.embedding(dfs_ids)).permute([1, 0, 2]).contiguous(), src_key_padding_mask=(source_mask - 1).bool())
            #dfs_encoder_output = dfs_outputs.contiguous()
            # - torch.mean(src_outputs[1:src_real_len[0] + 1, 0, :], dim=0)

            # compute contrastive loss
            src_tuple = tuple()
            pdg_tuple = tuple()

            src_real_length = src_real_len.cpu().numpy().tolist()
            pdg_real_length = pdg_real_len.cpu().numpy().tolist()
            '''
            batch_size = source_ids.size()[0]
            token_length = source_ids.size()[1]
            src_real_length = [1] * batch_size
            pdg_real_length = [1] * batch_size
            for i in range(batch_size):
                for j in range(1, token_length):
                    if source_ids[i, j] == 0:
                        # print(i, j, source_ids[i, j])
                        src_real_length[i] = j - 1
                    if source_ids[i, j] == 2:
                        # print(i, j, source_ids[i, j])
                        pdg_real_length[i] = j - src_real_length[i] - 2
                        break'''



            for i in range(len(src_real_length)):
                src_tuple += (torch.mean(src_encoder_output[1:src_real_length[i] + 1, i, :], dim=0).unsqueeze(0),)
            for i in range(len(pdg_real_length)):
                pdg_tuple += (torch.mean(src_encoder_output[2+src_real_length[i]: 2+src_real_length[i]+pdg_real_length[i], i, :], dim=0).unsqueeze(0),)
            src_context = torch.cat(src_tuple, dim=0)
            pdg_context = torch.cat(pdg_tuple, dim=0)

            #src_context = torch.mean(src_encoder_output.permute([1, 0, 2]).contiguous(), dim=1)
            #dfs_context = torch.mean(dfs_encoder_output.permute([1, 0, 2]).contiguous(), dim=1)
            contr_loss = self.constrativeLoss(src_context, pdg_context)

            attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            #tgt_embeddings = pe(self.embedding_nl(target_ids)).permute([1, 0, 2]).contiguous() # trans
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1, 0, 2]).contiguous()  # codebert
            src_out = self.decoder1(tgt_embeddings, src_encoder_output, tgt_mask=attn_mask,
                                    memory_key_padding_mask=(1 - source_mask).bool())

            #dfs_out = self.decoder1(tgt_embeddings, dfs_encoder_output, tgt_mask=attn_mask,memory_key_padding_mask=(1 - dfs_mask).bool())

            #rev_out = self.decoder1(tgt_embeddings, rev_encoder_output, tgt_mask=attn_mask, memory_key_padding_mask=(1 - rev_mask).bool())
            #decoder_output = self.decoder1(tgt_embeddings, encoder_output, tgt_mask=attn_mask, memory_key_padding_mask=(1 - source_mask).bool())
            src_hidden_states = src_out.permute([1, 0, 2]).contiguous()
            #dfs_hidden_states = dfs_out.permute([1, 0, 2]).contiguous()
            #rev_hidden_states = rev_out.permute([1, 0, 2]).contiguous()
            #hidden_states = torch.tanh(decoder_output.permute([1, 0, 2]).contiguous())

            #hidden_states = torch.tanh(self.cat_lin(torch.cat((src_hidden_states, dfs_hidden_states), 2)))
            hidden_states = torch.tanh(src_hidden_states)
            # src_hidden_states = torch.tanh(self.dense(src_out)).permute([1,0,2]).contiguous()
            # dfs_hidden_states = torch.tanh(self.dense(dfs_out)).permute([1,0,2]).contiguous()

            # hidden_states = torch.max(src_hidden_states, dfs_hidden_states)
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss*0.95 + contr_loss*0.05, loss * active_loss.sum(), active_loss.sum()
            return outputs
        else:
            # Predict
            preds = []
            #zero = torch.cuda.LongTensor(1).fill_(0)
            zero = torch.LongTensor(1).fill_(0).to(self.args.device)
            # print("source_ids.shape", source_ids.shape[0])
            for i in range(source_ids.shape[0]):
                src_context = src_encoder_output[:, i:i + 1]
                src_context_mask = source_mask[i:i + 1, :]

                #dfs_context = dfs_encoder_output[:, i:i + 1]
                #dfs_context_mask = dfs_mask[i:i + 1, :]

                #rev_context = rev_encoder_output[:, i:i + 1]
                #rev_context_mask = rev_mask[i:i + 1, :]
                """context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]"""

                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState().to(self.args.device)
                src_context = src_context.repeat(1, self.beam_size, 1)
                src_context_mask = src_context_mask.repeat(self.beam_size, 1)

                #dfs_context = dfs_context.repeat(1, self.beam_size, 1)
                #dfs_context_mask = dfs_context_mask.repeat(self.beam_size, 1)

                #rev_context = rev_context.repeat(1, self.beam_size, 1)
                #rev_context_mask = rev_context_mask.repeat(self.beam_size, 1)
                """context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)"""

                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])

                    #tgt_embeddings = pe(self.embedding_nl(input_ids)).permute([1, 0, 2]).contiguous()
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()  # codebert

                    src_out = self.decoder1(tgt_embeddings, src_context, tgt_mask=attn_mask,
                                            memory_key_padding_mask=(1 - src_context_mask).bool())
                    # src_out = torch.tanh(self.dense(src_out))
                    src_hidden_states = src_out.permute([1, 0, 2]).contiguous()[:, -1, :]

                    #dfs_out = self.decoder1(tgt_embeddings, dfs_context, tgt_mask=attn_mask,memory_key_padding_mask=(1 - dfs_context_mask).bool())
                    #dfs_hidden_states = dfs_out.permute([1, 0, 2]).contiguous()[:, -1, :]

                    #rev_out = self.decoder1(tgt_embeddings, rev_context, tgt_mask=attn_mask,memory_key_padding_mask=(1 - rev_context_mask).bool())
                    #rev_hidden_states = rev_out.permute([1, 0, 2]).contiguous()[:, -1, :]  # [1,768]
                    #decoder_output = self.decoder1(tgt_embeddings, context, tgt_mask=attn_mask, memory_key_padding_mask=(1 - context_mask).bool())
                    #hidden_states = torch.tanh(decoder_output.permute([1, 0, 2]).contiguous()[:, -1, :])
                    #hidden_states = torch.tanh(self.cat_lin(torch.cat((src_hidden_states, dfs_hidden_states), 1)))
                    hidden_states = torch.tanh(src_hidden_states)
                    # hidden_states = torch.max(src_hidden_states, dfs_hidden_states)
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                # print("pred -->", len(pred))
                # print("pred 0 ", len(pred[0]))
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]

                # print("pred ==>", len(pred))
                # print("pred 0 ==>", pred[0].size())
                preds.append(torch.cat(pred, 0).unsqueeze(0))
                # print("preds == >", len(preds))
                # print("preds 0 == >", preds[0].size())
            preds = torch.cat(preds, 0)
            # print("preds == >", preds.size())
            return preds


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
