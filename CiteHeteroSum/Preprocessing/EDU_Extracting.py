import re
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

global_gpu_id = 0
random_seed = 125
iter_display_size = 5

random_with_pre_shuffle = True
add_first_and_last = True
average_edu_level = True
document_enc_gru = True
enc_rnn_layer_num = 1

tree_infer_mode = "depth"
different_learning_rate = True
use_dev_set = True

dev_set_size = 15
hidden_size = 768

use_micro_F1 = True
use_org_Parseval = False

save_model = False

use_dwa_loss = True
if_edu_start_loss = True

os.environ["CUDA_VISIBLE_DEVICES"] = str(global_gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
From  DMRST tool: https://arxiv.org/pdf/2110.04518
'''
class EncoderRNN(nn.Module):
    def __init__(self, language_model, word_dim, hidden_size, rnn_layers, dropout, bert_tokenizer=None, segmenter=None):

        super(EncoderRNN, self).__init__()
        '''
        Input:
            [batch,length]
        Output: 
            encoder_output: [batch,length,hidden_size]    
            encoder_hidden: [rnn_layers,batch,hidden_size]
        '''

        self.rnn_layers = rnn_layers
        self.hidden_size = hidden_size
        self.word_dim = word_dim

        self.nnDropout = nn.Dropout(dropout)
        self.language_model = language_model

        self.layer_norm = nn.LayerNorm(word_dim, elementwise_affine=True)
        self.layer_norm_for_seg = nn.LayerNorm(word_dim, elementwise_affine=True)

        self.bert_tokenizer = bert_tokenizer
        self.reduce_dim_layer = nn.Linear(word_dim * 3, word_dim, bias=False)

        self.segmenter = segmenter
        self.doc_gru_enc = nn.GRU(word_dim, int(word_dim / 2), num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)

    def forward(self, input_sentence, EDU_breaks, is_test=False):
        if EDU_breaks is not None or is_test is False:
            max_edu_break_num = max([len(tmp_l) for tmp_l in EDU_breaks])
        all_outputs = []
        all_hidden = []

        batch_token_len_list = [len(i) for i in input_sentence]
        batch_token_len_max = max(batch_token_len_list)

        """ version 3.0 """
        total_edu_loss = torch.FloatTensor([0.0]).cuda()
        predict_edu_breaks_list = []
        tem_outputs = []

        """ For averaging the edu level embeddings START """
        for i in range(len(input_sentence)):
            bert_token_ids = [self.bert_tokenizer.convert_tokens_to_ids(input_sentence[i])]
            bert_token_ids = torch.LongTensor(bert_token_ids).cuda()

            """ fixed sliding window for encoding long sequence """
            window_size = 300
            sequence_length = len(input_sentence[i])
            slide_steps = int(np.ceil(len(input_sentence[i]) / window_size))
            window_embed_list = []
            for tmp_step in range(slide_steps):
                if tmp_step == 0:
                    one_win_res = self.language_model(bert_token_ids[:, :500])[0][:, :window_size, :]
                    window_embed_list.append(one_win_res)
                elif tmp_step == slide_steps - 1:
                    one_win_res = self.language_model(bert_token_ids[:, -((sequence_length - (window_size * tmp_step)) + 200):])[0][:, 200:, :]
                    window_embed_list.append(one_win_res)
                else:
                    one_win_res = self.language_model(bert_token_ids[:, (window_size * tmp_step - 100):(window_size * (tmp_step + 1) + 100)])[0][:, 100:400, :]
                    window_embed_list.append(one_win_res)

            embeddings = torch.cat(window_embed_list, dim=1)
            assert embeddings.size(1) == sequence_length
            embeddings = self.layer_norm(embeddings)

            """ add segmentation process """
            if is_test:
                predict_edu_breaks = self.segmenter.test_segment_loss(embeddings.squeeze())
                cur_edu_break = predict_edu_breaks
                predict_edu_breaks_list.append(predict_edu_breaks)

            else:
                cur_edu_break = EDU_breaks[i]
                seg_loss = self.segmenter.train_segment_loss(embeddings.squeeze(), cur_edu_break)
                """ Use this to pass the segmenation loss part: only for debug """
                # seg_loss = 0.0
                total_edu_loss += seg_loss

            # apply dropout
            embeddings = self.nnDropout(embeddings.squeeze(dim=0))
            tmp_average_list = []
            tmp_break_list = [0, ] + [tmp_j + 1 for tmp_j in cur_edu_break]
            for tmp_i in range(len(tmp_break_list) - 1):
                assert tmp_break_list[tmp_i] < tmp_break_list[tmp_i + 1]
                tmp_average_list.append(torch.mean(embeddings[tmp_break_list[tmp_i]:tmp_break_list[tmp_i + 1], :], dim=0, keepdim=True))
            tmp_average_embed = torch.cat(tmp_average_list, dim=0).unsqueeze(dim=0)
            outputs = tmp_average_embed

            """ For averaging the edu level embeddings END """
            if document_enc_gru is True:
                outputs, hidden = self.doc_gru_enc(outputs)
                hidden = hidden.view(2, 2, 1, int(self.word_dim / 2))[-1]
                hidden = hidden.transpose(0, 1).view(1, 1, -1).contiguous()

            if add_first_and_last is True:
                first_words = []
                last_words = []
                for tmp_i in range(len(tmp_break_list) - 1):
                    first_words.append(embeddings[tmp_break_list[tmp_i]].unsqueeze(dim=0))
                    last_words.append(embeddings[tmp_break_list[tmp_i + 1] - 1].unsqueeze(dim=0))

                outputs = torch.cat((outputs, torch.cat(first_words, dim=0).unsqueeze(dim=0), torch.cat(last_words, dim=0).unsqueeze(dim=0)), dim=2)
                outputs = self.reduce_dim_layer(outputs)

            tem_outputs.append(outputs)
            all_hidden.append(hidden)

        if is_test:
            max_edu_break_num = max([len(tmp_l) for tmp_l in predict_edu_breaks_list])
        for output in tem_outputs:
            cur_break_num = output.size(1)
            all_outputs.append(torch.cat([output, torch.zeros(1, max_edu_break_num - cur_break_num, self.word_dim).cuda()], dim=1))

        res_merged_output = torch.cat(all_outputs, dim=0)
        res_merged_hidden = torch.cat(all_hidden, dim=1)

        return res_merged_output, res_merged_hidden, total_edu_loss, predict_edu_breaks_list

    def GetEDURepresentation(self, input_sentence):
        tmp_max_token_num = len(input_sentence[0])
        bert_token_ids = [self.bert_tokenizer.convert_tokens_to_ids(v) + [5, ] * (tmp_max_token_num - len(v)) for k, v in enumerate(input_sentence)]
        bert_token_ids = torch.LongTensor(bert_token_ids).cuda()
        bert_embeddings = self.language_model(bert_token_ids)

        return bert_embeddings[0]

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_layers, dropout):
        super(DecoderRNN, self).__init__()

        '''
        Input:
            input: [1,length,input_size]
            initial_hidden_state: [rnn_layer,1,hidden_size]
        Output:
            output: [1,length,input_size]
            hidden_states: [rnn_layer,1,hidden_size]
        '''
        self.gru = nn.GRU(input_size, hidden_size, num_layers=rnn_layers, batch_first=True, dropout=(0 if rnn_layers == 1 else dropout))

    def forward(self, input_hidden_states, last_hidden):
        outputs, hidden = self.gru(input_hidden_states, last_hidden)

        return outputs, hidden


class PointerAtten(nn.Module):
    def __init__(self, atten_model, hidden_size):
        super(PointerAtten, self).__init__()

        '''       
        Input:
            Encoder_outputs: [length,encoder_hidden_size]
            Current_decoder_output: [decoder_hidden_size] 
            Attention_model: 'Biaffine' or 'Dotproduct' 

        Output:
            attention_weights: [1,length]
            log_attention_weights: [1,length]
        '''

        self.atten_model = atten_model
        self.weight1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.weight2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, cur_decoder_output):

        if self.atten_model == 'Biaffine':

            EW1_temp = self.weight1(encoder_outputs)
            EW1 = torch.matmul(EW1_temp, cur_decoder_output).unsqueeze(1)
            EW2 = self.weight2(encoder_outputs)
            bi_affine = EW1 + EW2
            bi_affine = bi_affine.permute(1, 0)

            atten_weights = F.softmax(bi_affine, 0)
            log_atten_weights = F.log_softmax(bi_affine + 1e-6, 0)

        elif self.atten_model == 'Dotproduct':

            dot_prod = torch.matmul(encoder_outputs, cur_decoder_output).unsqueeze(0)
            atten_weights = F.softmax(dot_prod, 1)
            log_atten_weights = F.log_softmax(dot_prod + 1e-6, 1)

        return atten_weights, log_atten_weights


class LabelClassifier(nn.Module):
    def __init__(self, input_size, classifier_hidden_size, classes_label=41,
                 bias=True, dropout=0.5):

        super(LabelClassifier, self).__init__()
        '''

        Args:
            input_size: input size
            classifier_hidden_size: project input to classifier space
            classes_label: corresponding to 39 relations we have. 
                           (e.g. Contrast_NN)
            bias: If set to False, the layer will not learn an additive bias.
                Default: True               

        Input:
            input_left: [1,input_size]
            input_right: [1,input_size]
        Output:
            relation_weights: [1,classes_label]
            log_relation_weights: [1,classes_label]

        '''
        self.classifier_hidden_size = classifier_hidden_size
        self.labelspace_left = nn.Linear(input_size, classifier_hidden_size, bias=False)
        self.labelspace_right = nn.Linear(input_size, classifier_hidden_size, bias=False)
        self.weight_left = nn.Linear(classifier_hidden_size, classes_label, bias=False)
        self.weight_right = nn.Linear(classifier_hidden_size, classes_label, bias=False)
        self.nnDropout = nn.Dropout(dropout)

        self.classifier_hidden_size = classifier_hidden_size

        if bias:
            self.weight_bilateral = nn.Bilinear(classifier_hidden_size, classifier_hidden_size, classes_label)
        else:
            self.weight_bilateral = nn.Bilinear(classifier_hidden_size, classifier_hidden_size, classes_label,
                                                bias=False)

    def forward(self, input_left, input_right):

        left_size = input_left.size()
        right_size = input_right.size()

        labelspace_left = F.elu(self.labelspace_left(input_left))
        labelspace_right = F.elu(self.labelspace_right(input_right))

        # Apply dropout
        union = torch.cat((labelspace_left, labelspace_right), 1)
        union = self.nnDropout(union)
        labelspace_left = union[:, :self.classifier_hidden_size]
        labelspace_right = union[:, self.classifier_hidden_size:]

        output = (self.weight_bilateral(labelspace_left, labelspace_right) +
                  self.weight_left(labelspace_left) + self.weight_right(labelspace_right))

        # Obtain relation weights and log relation weights (for loss)
        relation_weights = F.softmax(output, 1)
        log_relation_weights = F.log_softmax(output + 1e-6, 1)

        return relation_weights, log_relation_weights

class Segmenter_pointer(nn.Module):

    def __init__(self, hidden_size, atten_model=None, decoder_input_size=None, rnn_layers=None, dropout_d=None):
        super(Segmenter_pointer, self).__init__()

        self.hidden_size = hidden_size
        self.pointer = PointerAtten(atten_model, hidden_size)
        self.encoder = nn.GRU(hidden_size, int(hidden_size / 2), num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
        self.decoder = DecoderRNN(decoder_input_size, hidden_size, rnn_layers, dropout_d)
        self.loss_function = nn.NLLLoss()

    def forward(self):
        raise RuntimeError('Segmenter does not have forward process.')

    def train_segment_loss(self, word_embeddings, edu_breaks):
        outputs, last_hidden = self.encoder(word_embeddings.unsqueeze(0))
        outputs = outputs.squeeze()
        cur_decoder_hidden = outputs[-1, :].unsqueeze(0).unsqueeze(0)
        edu_breaks = [0] + edu_breaks
        total_loss = torch.FloatTensor([0.0]).cuda()
        for step, start_index in enumerate(edu_breaks[:-1]):
            cur_decoder_output, cur_decoder_hidden = self.decoder(outputs[start_index].unsqueeze(0).unsqueeze(0), last_hidden=cur_decoder_hidden)

            _, log_atten_weights = self.pointer(outputs[start_index:], cur_decoder_output.squeeze(0).squeeze(0))
            cur_ground_index = torch.tensor([edu_breaks[step + 1] - start_index]).cuda()
            total_loss = total_loss + self.loss_function(log_atten_weights, cur_ground_index)

        return total_loss

    def test_segment_loss(self, word_embeddings, edu_breaks):
        outputs, last_hidden = self.encoder(word_embeddings.unsqueeze(0))
        outputs = outputs.squeeze()
        cur_decoder_hidden = outputs[-1, :].unsqueeze(0).unsqueeze(0)
        start_index = 0
        predict_segment = []
        sentence_length = outputs.shape[0]
        while start_index < sentence_length:
            cur_decoder_output, cur_decoder_hidden = self.decoder(outputs[start_index].unsqueeze(0).unsqueeze(0), last_hidden=cur_decoder_hidden)
            atten_weights, log_atten_weights = self.pointer(outputs[start_index:], cur_decoder_output.squeeze(0).squeeze(0))
            _, top_index_seg = atten_weights.topk(1)

            seg_index = int(top_index_seg[0][0]) + start_index
            predict_segment.append(seg_index)
            start_index = seg_index + 1

        if predict_segment[-1] != sentence_length - 1:
            predict_segment.append(sentence_length - 1)

        return predict_segment

class Segmenter(nn.Module):
    def __init__(self, hidden_size):
        super(Segmenter, self).__init__()

        self.hidden_size = hidden_size
        self.drop_out = nn.Dropout(p=0.5)
        self.linear = nn.Linear(hidden_size, 2)
        self.linear_start = nn.Linear(hidden_size, 2)
        self.loss_function = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 10.0]).cuda())

    def forward(self):
        raise RuntimeError('Segmenter does not have forward process.')

    def train_segment_loss(self, word_embeddings, edu_breaks):
        edu_break_target = [0, ] * word_embeddings.size(0)
        edu_start_target = [0, ] * word_embeddings.size(0)

        for i in edu_breaks:
            edu_break_target[i] = 1
        edu_start_target[0] = 1
        for i in edu_breaks[:-1]:
            edu_start_target[i + 1] = 1

        edu_break_target = torch.LongTensor(edu_break_target).cuda()
        edu_start_target = torch.LongTensor(edu_start_target).cuda()
        outputs = self.linear(self.drop_out(word_embeddings))
        start_outputs = self.linear_start(self.drop_out(word_embeddings))

        if if_edu_start_loss:
            total_loss = self.loss_function(outputs, edu_break_target) + self.loss_function(start_outputs, edu_start_target)
        else:
            total_loss = self.loss_function(outputs, edu_break_target)
        return total_loss

    def test_segment_loss(self, word_embeddings):
        outputs = self.linear(self.drop_out(word_embeddings))
        pred = torch.argmax(outputs, dim=1).detach()
        predict_segment = [i for i, k in enumerate(pred) if k == 1]

        if word_embeddings.size(0) - 1 not in predict_segment:
            predict_segment.append(word_embeddings.size(0) - 1)

        return predict_segment


def get_RelationAndNucleus(label_index):
    RelationTable = ['Attribution_SN', 'Enablement_NS', 'Cause_SN', 'Cause_NN', 'Temporal_SN',
                     'Condition_NN', 'Cause_NS', 'Elaboration_NS', 'Background_NS',
                     'Topic-Comment_SN', 'Elaboration_SN', 'Evaluation_SN', 'Explanation_NN',
                     'TextualOrganization_NN', 'Background_SN', 'Contrast_NN', 'Evaluation_NS',
                     'Topic-Comment_NN', 'Condition_NS', 'Comparison_NS', 'Explanation_SN',
                     'Contrast_NS', 'Comparison_SN', 'Condition_SN', 'Summary_SN', 'Explanation_NS',
                     'Enablement_SN', 'Temporal_NN', 'Temporal_NS', 'Topic-Comment_NS',
                     'Manner-Means_NS', 'Same-Unit_NN', 'Summary_NS', 'Contrast_SN',
                     'Attribution_NS', 'Manner-Means_SN', 'Joint_NN', 'Comparison_NN', 'Evaluation_NN',
                     'Topic-Change_NN', 'Topic-Change_NS', 'Summary_NN', ]

    relation = RelationTable[label_index]
    temp = re.split(r'_', relation)
    sub1 = temp[0]
    sub2 = temp[1]

    if sub2 == 'NN':
        Nuclearity_left = 'Nucleus'
        Nuclearity_right = 'Nucleus'
        Relation_left = sub1
        Relation_right = sub1

    elif sub2 == 'NS':
        Nuclearity_left = 'Nucleus'
        Nuclearity_right = 'Satellite'
        Relation_left = 'span'
        Relation_right = sub1

    elif sub2 == 'SN':
        Nuclearity_left = 'Satellite'
        Nuclearity_right = 'Nucleus'
        Relation_left = sub1
        Relation_right = 'span'

    return Nuclearity_left, Nuclearity_right, Relation_left, Relation_right

class ParsingNet(nn.Module):
    def __init__(self, language_model, word_dim=768, hidden_size=768, decoder_input_size=768,
                 atten_model="Dotproduct", classifier_input_size=768, classifier_hidden_size=768, classes_label=42, classifier_bias=True,
                 rnn_layers=1, dropout_e=0.5, dropout_d=0.5, dropout_c=0.5, bert_tokenizer=None):

        super(ParsingNet, self).__init__()
        '''
        Args:
            batch_size: batch size
            word_dim: word embedding dimension 
            hidden_size: hidden size of encoder and decoder 
            decoder_input_size: input dimension of decoder
            atten_model: pointer attention machanisam, 'Dotproduct' or 'Biaffine' 
            device: device that our model is running on 
            classifier_input_size: input dimension of labels classifier 
            classifier_hidden_size: classifier hidden space
            classes_label: relation(label) number, default = 39
            classifier_bias: bilinear bias in classifier, default = True
            rnn_layers: encoder and decoder layer number
            dropout: dropout rate
        '''
        self.word_dim = word_dim
        self.hidden_size = hidden_size
        self.decoder_input_size = decoder_input_size
        self.classifier_input_size = classifier_input_size
        self.classifier_hidden_size = classifier_hidden_size
        self.classes_label = classes_label
        self.classifier_bias = classifier_bias
        self.rnn_layers = rnn_layers
        self.segmenter = Segmenter(hidden_size)
        self.encoder = EncoderRNN(language_model, word_dim, hidden_size, enc_rnn_layer_num, dropout_e, bert_tokenizer=bert_tokenizer, segmenter=self.segmenter)
        self.decoder = DecoderRNN(decoder_input_size, hidden_size, rnn_layers, dropout_d)
        self.pointer = PointerAtten(atten_model, hidden_size)
        self.getlabel = LabelClassifier(classifier_input_size, classifier_hidden_size, classes_label, bias=True, dropout=dropout_c)

    def forward(self):
        raise RuntimeError('Parsing Network does not have forward process.')

    def TrainingLoss(self, input_sentence, EDU_breaks, LabelIndex, ParsingIndex, DecoderInputIndex, ParentsIndex, SiblingIndex):

        # Obtain encoder outputs and last hidden states
        EncoderOutputs, Last_Hiddenstates, total_edu_loss, _ = self.encoder(input_sentence, EDU_breaks)

        Label_LossFunction = nn.NLLLoss()
        Span_LossFunction = nn.NLLLoss()

        Loss_label_batch = 0
        Loss_tree_batch = torch.FloatTensor([0.0]).cuda()
        Loop_label_batch = 0
        Loop_tree_batch = 0

        batch_size = len(LabelIndex)
        for i in range(batch_size):

            cur_LabelIndex = LabelIndex[i]
            cur_LabelIndex = torch.tensor(cur_LabelIndex)
            cur_LabelIndex = cur_LabelIndex.cuda()
            cur_ParsingIndex = ParsingIndex[i]
            cur_DecoderInputIndex = DecoderInputIndex[i]
            cur_ParentsIndex = ParentsIndex[i]
            cur_SiblingIndex = SiblingIndex[i]

            if len(EDU_breaks[i]) == 1:

                continue

            elif len(EDU_breaks[i]) == 2:

                # Obtain the encoded representations. The dimension: [2,hidden_size]
                cur_EncoderOutputs = EncoderOutputs[i][:len(EDU_breaks[i])]

                # Use the last hidden state of a span to predict the relation between these two span.
                input_left = cur_EncoderOutputs[0].unsqueeze(0)
                input_right = cur_EncoderOutputs[1].unsqueeze(0)

                _, log_relation_weights = self.getlabel(input_left, input_right)

                Loss_label_batch = Loss_label_batch + Label_LossFunction(log_relation_weights, cur_LabelIndex)
                Loop_label_batch = Loop_label_batch + 1

            else:
                cur_EncoderOutputs = EncoderOutputs[i][:len(EDU_breaks[i])]
                cur_Last_Hiddenstates = Last_Hiddenstates[:, i, :].unsqueeze(1)
                cur_decoder_hidden = cur_Last_Hiddenstates.contiguous()

                EDU_index = [x for x in range(len(cur_EncoderOutputs))]
                stacks = ['__StackRoot__', EDU_index]

                for j in range(len(cur_DecoderInputIndex)):

                    if stacks[-1] != '__StackRoot__':
                        stack_head = stacks[-1]

                        if len(stack_head) < 3:

                            input_left = cur_EncoderOutputs[cur_ParsingIndex[j]].unsqueeze(0)
                            input_right = cur_EncoderOutputs[stack_head[-1]].unsqueeze(0)

                            assert cur_ParsingIndex[j] < stack_head[-1]

                            cur_decoder_input = torch.mean(cur_EncoderOutputs[stack_head], keepdim=True, dim=0).unsqueeze(0)
                            cur_decoder_output, cur_decoder_hidden = self.decoder(cur_decoder_input, last_hidden=cur_decoder_hidden)

                            _, log_relation_weights = self.getlabel(input_left, input_right)
                            Loss_label_batch = Loss_label_batch + Label_LossFunction(log_relation_weights, cur_LabelIndex[j].unsqueeze(0))

                            del stacks[-1]
                            Loop_label_batch = Loop_label_batch + 1

                        else: 

                            cur_decoder_input = torch.mean(cur_EncoderOutputs[stack_head], keepdim=True, dim=0).unsqueeze(0)

                            cur_decoder_output, cur_decoder_hidden = self.decoder(cur_decoder_input, last_hidden=cur_decoder_hidden)

                            _, log_atten_weights = self.pointer(cur_EncoderOutputs[stack_head[:-1]], cur_decoder_output.squeeze(0).squeeze(0))
                            cur_ground_index = torch.tensor([int(cur_ParsingIndex[j]) - int(stack_head[0])])
                            cur_ground_index = cur_ground_index.cuda()
                            Loss_tree_batch = Loss_tree_batch + Span_LossFunction(log_atten_weights, cur_ground_index)

                            """ merge edu level representation for left and right siblings START """
                            if average_edu_level is True:
                                input_left = torch.mean(cur_EncoderOutputs[stack_head[0]:cur_ParsingIndex[j] + 1, :], keepdim=True, dim=0)
                                input_right = torch.mean(cur_EncoderOutputs[cur_ParsingIndex[j] + 1: stack_head[-1] + 1, :], keepdim=True, dim=0)
                            else:
                                input_left = cur_EncoderOutputs[cur_ParsingIndex[j]].unsqueeze(0)
                                input_right = cur_EncoderOutputs[stack_head[-1]].unsqueeze(0)
                            """ merge edu level representation for left and right siblings END """

                            _, log_relation_weights = self.getlabel(input_left, input_right)
                            Loss_label_batch = Loss_label_batch + Label_LossFunction(log_relation_weights, cur_LabelIndex[j].unsqueeze(0))

                            stack_left = stack_head[:(cur_ParsingIndex[j] - stack_head[0] + 1)]
                            stack_right = stack_head[(cur_ParsingIndex[j] - stack_head[0] + 1):]
                            del stacks[-1]
                            Loop_label_batch = Loop_label_batch + 1
                            Loop_tree_batch = Loop_tree_batch + 1

                            if len(stack_right) > 1:
                                stacks.append(stack_right)
                            if len(stack_left) > 1:
                                stacks.append(stack_left)

        Loss_label_batch = Loss_label_batch / Loop_label_batch

        if Loop_tree_batch == 0:
            Loop_tree_batch = 1

        Loss_tree_batch = Loss_tree_batch / Loop_tree_batch

        return Loss_tree_batch, Loss_label_batch, total_edu_loss


    def TestingLoss(self, input_sentence, input_EDU_breaks, LabelIndex, ParsingIndex, GenerateTree, use_pred_segmentation):
        '''
            Input:
                input_sentence: [batch_size, length]
                input_EDU_breaks: e.g. [[2,4,6,9],[2,5,8,10,13],[6,8],[6]]
                LabelIndex: e.g. [[0,3,32],[20,11,14,19],[20],[],]
                ParsingIndex: e.g. [[1,2,0],[3,2,0,1],[0],[]]
            Output: log_atten_weights
                Average loss of tree in a batch
                Average loss of relation in a batch
        '''
        EncoderOutputs, Last_Hiddenstates, _, predict_edu_breaks = self.encoder(input_sentence, input_EDU_breaks, is_test=use_pred_segmentation)

        if use_pred_segmentation:
            EDU_breaks = predict_edu_breaks
            if LabelIndex is None and ParsingIndex is None:
                LabelIndex = [[0, ] * (len(i) - 1) for i in EDU_breaks]
                ParsingIndex = [[0, ] * (len(i) - 1) for i in EDU_breaks]
        else:
            EDU_breaks = input_EDU_breaks

        Label_LossFunction = nn.NLLLoss()
        Span_LossFunction = nn.NLLLoss()

        Loss_label_batch = torch.FloatTensor([0.0]).cuda()
        Loss_tree_batch = torch.FloatTensor([0.0]).cuda()
        Loop_label_batch = 0
        Loop_tree_batch = 0

        Label_batch = []
        Tree_batch = []

        if GenerateTree:
            SPAN_batch = []

        for i in range(len(EDU_breaks)):

            cur_label = []
            cur_tree = []

            cur_LabelIndex = LabelIndex[i]
            cur_LabelIndex = torch.tensor(cur_LabelIndex)
            cur_LabelIndex = cur_LabelIndex.cuda()
            cur_ParsingIndex = ParsingIndex[i]

            if len(EDU_breaks[i]) == 1:
                Tree_batch.append([])
                Label_batch.append([])

                if GenerateTree:
                    SPAN_batch.append(['NONE'])

            elif len(EDU_breaks[i]) == 2:

                cur_EncoderOutputs = EncoderOutputs[i][:len(EDU_breaks[i])]

                input_left = cur_EncoderOutputs[0].unsqueeze(0)
                input_right = cur_EncoderOutputs[1].unsqueeze(0)
                relation_weights, log_relation_weights = self.getlabel(input_left, input_right)
                _, topindex = relation_weights.topk(1)
                LabelPredict = int(topindex[0][0])
                Tree_batch.append([0])
                Label_batch.append([LabelPredict])

                if use_pred_segmentation is False:
                    Loss_label_batch = Loss_label_batch + Label_LossFunction(log_relation_weights, cur_LabelIndex)

                Loop_label_batch = Loop_label_batch + 1

                if GenerateTree:
                    Nuclearity_left, Nuclearity_right, Relation_left, Relation_right = get_RelationAndNucleus(LabelPredict)

                    Span = '(1:' + str(Nuclearity_left) + '=' + str(Relation_left) + \
                           ':1,2:' + str(Nuclearity_right) + '=' + str(Relation_right) + ':2)'
                    SPAN_batch.append([Span])

            else:
                cur_EncoderOutputs = EncoderOutputs[i][:len(EDU_breaks[i])]

                EDU_index = [x for x in range(len(cur_EncoderOutputs))]
                stacks = ['__StackRoot__', EDU_index]

                cur_Last_Hiddenstates = Last_Hiddenstates[:, i, :].unsqueeze(1)
                cur_decoder_hidden = cur_Last_Hiddenstates.contiguous()

                LoopIndex = 0

                if GenerateTree:
                    Span = ''

                tmp_decode_step = -1

                while stacks[-1] != '__StackRoot__':
                    stack_head = stacks[-1]

                    if len(stack_head) < 3:

                        tmp_decode_step += 1
                        input_left = cur_EncoderOutputs[stack_head[0]].unsqueeze(0)
                        input_right = cur_EncoderOutputs[stack_head[-1]].unsqueeze(0)

                        relation_weights, log_relation_weights = self.getlabel(input_left, input_right)
                        _, topindex = relation_weights.topk(1)
                        LabelPredict = int(topindex[0][0])
                        cur_label.append(LabelPredict)

                        cur_tree.append(stack_head[0])
                        cur_decoder_input = torch.mean(cur_EncoderOutputs[stack_head], keepdim=True, dim=0).unsqueeze(0)
                        cur_decoder_output, cur_decoder_hidden = self.decoder(cur_decoder_input, last_hidden=cur_decoder_hidden)

                        if LoopIndex > (len(cur_ParsingIndex) - 1):
                            cur_Label_true = cur_LabelIndex[-1]
                        else:
                            cur_Label_true = cur_LabelIndex[LoopIndex]

                        if use_pred_segmentation is False:
                            Loss_label_batch = Loss_label_batch + Label_LossFunction(log_relation_weights, cur_Label_true.unsqueeze(0))

                        Loop_label_batch = Loop_label_batch + 1
                        LoopIndex = LoopIndex + 1
                        del stacks[-1]

                        if GenerateTree:
                            Nuclearity_left, Nuclearity_right, Relation_left, Relation_right = get_RelationAndNucleus(LabelPredict)

                            cur_span = '(' + str(stack_head[0] + 1) + ':' + str(Nuclearity_left) + '=' + str(Relation_left) + \
                                       ':' + str(stack_head[0] + 1) + ',' + str(stack_head[-1] + 1) + ':' + str(Nuclearity_right) + '=' + \
                                       str(Relation_right) + ':' + str(stack_head[-1] + 1) + ')'

                            Span = Span + ' ' + cur_span

                    else:  

                        tmp_decode_step += 1
                        cur_decoder_input = torch.mean(cur_EncoderOutputs[stack_head], keepdim=True, dim=0).unsqueeze(0)
                        cur_decoder_output, cur_decoder_hidden = self.decoder(cur_decoder_input, last_hidden=cur_decoder_hidden)
                        atten_weights, log_atten_weights = self.pointer(cur_EncoderOutputs[stack_head[:-1]], cur_decoder_output.squeeze(0).squeeze(0))

                        _, topindex_tree = atten_weights.topk(1)
                        TreePredict = int(topindex_tree[0][0]) + stack_head[0]

                        cur_tree.append(TreePredict)

                        """ merge edu level representation for left and right siblings START """
                        if average_edu_level is True:
                            input_left = torch.mean(cur_EncoderOutputs[stack_head[0]:TreePredict + 1, :], keepdim=True, dim=0)
                            input_right = torch.mean(cur_EncoderOutputs[TreePredict + 1: stack_head[-1] + 1, :], keepdim=True, dim=0)
                        else:
                            input_left = cur_EncoderOutputs[TreePredict].unsqueeze(0)
                            input_right = cur_EncoderOutputs[stack_head[-1]].unsqueeze(0)
                        """ merge edu level representation for left and right siblings END """

                        relation_weights, log_relation_weights = self.getlabel(input_left, input_right)
                        _, topindex_label = relation_weights.topk(1)
                        LabelPredict = int(topindex_label[0][0])
                        cur_label.append(LabelPredict)

                        if LoopIndex > (len(cur_ParsingIndex) - 1):
                            cur_Label_true = cur_LabelIndex[-1]
                            cur_Tree_true = cur_ParsingIndex[-1]
                        else:
                            cur_Label_true = cur_LabelIndex[LoopIndex]
                            cur_Tree_true = cur_ParsingIndex[LoopIndex]

                        temp_ground = max(0, (int(cur_Tree_true) - int(stack_head[0])))
                        if temp_ground >= (len(stack_head) - 1):
                            temp_ground = stack_head[-2] - stack_head[0]
                        cur_ground_index = torch.tensor([temp_ground])
                        cur_ground_index = cur_ground_index.cuda()

                        if use_pred_segmentation is False:
                            Loss_tree_batch = Loss_tree_batch + Span_LossFunction(log_atten_weights, cur_ground_index)
                            Loss_label_batch = Loss_label_batch + Label_LossFunction(log_relation_weights, cur_Label_true.unsqueeze(0))

                        stack_left = stack_head[:(TreePredict - stack_head[0] + 1)]
                        stack_right = stack_head[(TreePredict - stack_head[0] + 1):]

                        del stacks[-1]
                        Loop_label_batch = Loop_label_batch + 1
                        Loop_tree_batch = Loop_tree_batch + 1
                        LoopIndex = LoopIndex + 1

                        if len(stack_right) > 1:
                            stacks.append(stack_right)
                        if len(stack_left) > 1:
                            stacks.append(stack_left)

                        if GenerateTree:
                            Nuclearity_left, Nuclearity_right, Relation_left, Relation_right = \
                                get_RelationAndNucleus(LabelPredict)

                            cur_span = '(' + str(stack_head[0] + 1) + ':' + str(Nuclearity_left) + '=' + str(Relation_left) + \
                                       ':' + str(TreePredict + 1) + ',' + str(TreePredict + 2) + ':' + str(Nuclearity_right) + '=' + \
                                       str(Relation_right) + ':' + str(stack_head[-1] + 1) + ')'
                            Span = Span + ' ' + cur_span

                Tree_batch.append(cur_tree)
                Label_batch.append(cur_label)
                if GenerateTree:
                    SPAN_batch.append([Span.strip()])

        if Loop_label_batch == 0:
            Loop_label_batch = 1

        Loss_label_batch = Loss_label_batch / Loop_label_batch

        if Loop_tree_batch == 0:
            Loop_tree_batch = 1

        Loss_tree_batch = Loss_tree_batch / Loop_tree_batch

        Loss_label_batch = Loss_label_batch.detach()
        Loss_tree_batch = Loss_tree_batch.detach()

        merged_label_gold = []
        for tmp_i in LabelIndex:
            merged_label_gold.extend(tmp_i)

        merged_label_pred = []
        for tmp_i in Label_batch:
            merged_label_pred.extend(tmp_i)

        return Loss_tree_batch, Loss_label_batch, (SPAN_batch if GenerateTree else None), (merged_label_gold, merged_label_pred), EDU_breaks


def extract_spans(input_tokens, edu_boundaries):
    spans = []
    start_idx = 0

    for i, end_idx in enumerate(edu_boundaries):
        if i == len(edu_boundaries) - 1:
            span_tokens = input_tokens[start_idx:]
        else:
            span_tokens = input_tokens[start_idx:end_idx]
        span_text = ''.join(span_tokens).replace('▁', ' ').strip()
        spans.append(span_text)
        start_idx = end_idx

    return spans

def inference(model, tokenizer, input_sentences, batch_size):
    LoopNeeded = int(np.ceil(len(input_sentences) / batch_size))

    input_sentences = [tokenizer.tokenize(i, add_special_tokens=False) for i in input_sentences]
    all_segmentation_pred = []
    all_tree_parsing_pred = []

    with torch.no_grad():
        for loop in range(LoopNeeded):
            StartPosition = loop * batch_size
            EndPosition = (loop + 1) * batch_size
            if EndPosition > len(input_sentences):
                EndPosition = len(input_sentences)

            input_sen_batch = input_sentences[StartPosition:EndPosition]
            _, _, SPAN_batch, _, predict_EDU_breaks = model.TestingLoss(input_sen_batch, input_EDU_breaks=None, LabelIndex=None,
                                                                        ParsingIndex=None, GenerateTree=True, use_pred_segmentation=True)
            all_segmentation_pred.extend(predict_EDU_breaks)
            all_tree_parsing_pred.extend(SPAN_batch)
    return input_sentences, all_segmentation_pred, all_tree_parsing_pred

from transformers import AutoTokenizer, AutoModel

def load_model(model_path):
    bert_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
    bert_model = AutoModel.from_pretrained("xlm-roberta-base")
    bert_model = bert_model.to(device)

    for name, param in bert_model.named_parameters():
        param.requires_grad = False
    EDU_extract_model = ParsingNet(bert_model, bert_tokenizer=bert_tokenizer)
    EDU_extract_model = EDU_extract_model.to(device)

    state_dict = torch.load(model_path)
    state_dict.pop("encoder.language_model.embeddings.position_ids", None)
    EDU_extract_model.load_state_dict(state_dict)
    EDU_extract_model = EDU_extract_model.eval()
    return bert_tokenizer, EDU_extract_model
