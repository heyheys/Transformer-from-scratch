# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 20:05:11 2021

@author: fujye
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class transformer(nn.Module):
    def __init__(self, emb_size, src_voc_size, tgt_voc_size, attention_heads, encoder_num_stack, decoder_num_stack):
        '''        
        Parameters
        ----------
        emb_size : integer, must be a multiple of attention_heads.
            Number of dimensions for the word embeddings. 
            Here assume same embedding size for source and target sentences.
        src_voc_size : integer
            Vocabulary size of source language. Used for construct word embedding layers.
        tgt_voc_size : integer
            Vocabulary size of target language. Used for construct word embedding layers.
        attention_heads: integer.
            Here assumes the Encoder and Decoder have same number of attention_heads.
        encoder_num_stack: integer.
            The number of stacks in the Encoder.
        decoder_num_stack: integer.
            The number of stacks in the Decoder.
        '''
        
        super().__init__()
        self.emb_size = emb_size
        self.attention_heads = attention_heads
        self.encoder_num_stack = encoder_num_stack
        self.decoder_num_stack = decoder_num_stack
        
# =============================================================================
#         # common layers
# =============================================================================
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=.1)
        
        # ?? positioning layer (yet to do)
        
        self.layernorm = nn.LayerNorm(self.emb_size)
        
        self.last_affine = nn.Linear(self.emb_size, tgt_voc_size)  
                                     
# =============================================================================
#         ### construct Encoder layers        
# =============================================================================
        # input embedding layer
        self.src_emb = nn.Embedding(src_voc_size, emb_size,padding_idx=0) 
        
        # weight matrices for query, key and value
        self.encoder_QKV_matrices = {}
        for i in range(self.attention_heads):
            self.encoder_QKV_matrices['WK'+str(i)] = nn.Linear(emb_size, int(emb_size/attention_heads), bias=False)   # key weight matrix
            self.encoder_QKV_matrices['WQ'+str(i)] = nn.Linear(emb_size, int(emb_size/attention_heads), bias=False)   # query weight matrix
            self.encoder_QKV_matrices['WV'+str(i)] = nn.Linear(emb_size, int(emb_size/attention_heads), bias=False)   # value weight matrix

        # Encoder's feed-forward layers
        self.en_affine = {}
        self.en_affine['Wa0'] = nn.Linear(self.emb_size, self.emb_size)    
        for i in range(1, self.encoder_num_stack):
            self.en_affine['Wa'+str(i)] = nn.Linear(self.emb_size, self.emb_size)    
            self.en_affine['Wb'+str(i-1)] = nn.Linear(self.emb_size, self.emb_size)   
        self.en_affine['Wb'+str(i)] = nn.Linear(self.emb_size, self.emb_size)    
        # the hidden size of feed-forward layers must be equal to emb_size so that the residual can add. 
                
# =============================================================================
#         ### construct Decoder layers        
# =============================================================================
        # output embedding layer
        self.tgt_emb = nn.Embedding(tgt_voc_size, emb_size,padding_idx=0) 
        
        # weight matrices for query, key and value
        self.decoder_QKV_matrices = {}
        for i in range(self.attention_heads):
            self.decoder_QKV_matrices['WK'+str(i)] = nn.Linear(emb_size, int(emb_size/attention_heads), bias=False)   # key weight matrix
            self.decoder_QKV_matrices['WQ'+str(i)] = nn.Linear(emb_size, int(emb_size/attention_heads), bias=False)   # query weight matrix
            self.decoder_QKV_matrices['WV'+str(i)] = nn.Linear(emb_size, int(emb_size/attention_heads), bias=False)   # value weight matrix        

        # Decoder's feed-forward layers
        self.de_affine = {}
        self.de_affine['Wa0'] = nn.Linear(self.emb_size, self.emb_size)    
        for i in range(1, self.decoder_num_stack):
            self.de_affine['Wa'+str(i)] = nn.Linear(self.emb_size, self.emb_size)    
            self.de_affine['Wb'+str(i-1)] = nn.Linear(self.emb_size, self.emb_size)   
        self.de_affine['Wb'+str(i)] = nn.Linear(self.emb_size, self.emb_size)    
                
        
    def forward(self, src_sentence_batch, tgt_sentence_batch):
        '''
        Parameters
        ----------
        src_sentence_batch : tensor of shape (N, Tx), where N is the batch_size, Tx is the number of words in the source sentence. 
            These source sentences are already padded 0 if their lengths are less than the maximum length.                    
            Each value of a sentence is an integer mapping to the word in the vocabulary (e.g. [2, 4, 1]).            
        tgt_sentence_batch : tensor of shape (N, Ty), where N is the batch_size, Ty is the number of words in the target sentence. 
            These target sentences are already padded 0 if their lengths are less than the maximum length.                    
            Each value of a sentence is an integer mapping to the word in the vocabulary (e.g. [2, 4, 1]).

        Returns
        -------
        output : tensor of shape (N, Ty, tgt_voc_size).
            The output of Transformer.

        '''
        
        # go through the input embedding layer
        X = self.src_emb(src_sentence_batch)    # shape (N, Tx, emb_size). X is embeddings.
        Y = self.tgt_emb(tgt_sentence_batch)    # shape (N, Tx, emb_size). X is embeddings.
        
        # go through Encoder's stacks
        for stack in range(self.encoder_num_stack):
            X = self._Encoder_one_stack(X, stack)
        # now X is the Encoder's output. shape (N, Tx, emb_size).
        
        # go through Decoder's stacks
        for stack in range(self.decoder_num_stack):
            Y = self._Decoder_one_stack(Y, X, stack)       # shape (N, Ty, emb_size).
        
        # go through the last linear layer and softmax
        output = self.last_affine(Y)                       # shape (N, Ty, tgt_voc_size).
        output = F.softmax(output)                         # shape (N, Ty, tgt_voc_size).
        
        return output
        
        
    def _self_attention_step(self, X, module = 'Encoder'):
        '''
        Perform self-MultiHead-attention mechanism once. 
        This is called by _Encoder_one_stack() and should not be called manually.
        
        Parameters
        ----------
        X: tensor of shape (N, T, emb_size). Note that the sentence lengths of source sentences and target sentences are different.
            The previous embeddings (the output before this MultiHead attention layer)            
        module: 'Encoder' or 'Decoder'.
            If 'Encoder', masks will not be applied, and the query, key, value matrices are those for the Encoder.
            If 'Decoder', masks will be applied, and the query, key, value matrices are those for the Decoder. 
        
        Returns
        -------
        residual: tensor of shape (N, T, emb_size).
            Is the output of multi_head attention which can be thought as a refined word embeddings.
        
        '''
                
        # compute XK, XV, XQ (multipule heads), similarity (between XK and XQ), finally output
        outputs = []
        N, T, emb_size = X.size()
        
        if module == 'Encoder':            
            for i in range(self.attention_heads):
                XK_head = self.encoder_QKV_matrices['WK'+str(i)](X)                  # shape (N, T, emb_size/attention_heads)
                XQ_head = self.encoder_QKV_matrices['WQ'+str(i)](X)                  # shape (N, T, emb_size/attention_heads)
                similarity = torch.bmm(XK_head, XQ_head.permute(0, 2, 1))          # shape (N, T, T)
                alpha = F.softmax(similarity/np.sqrt(self.emb_size/self.attention_heads), dim=2)  # softmax and scaled dot product. shape (N, T, T)
                XV_head = self.encoder_QKV_matrices['WV'+str(i)](X)                  # shape (N, T, emb_size/attention_heads)
                outputs.append(torch.bmm(alpha, XV_head))                            # shape (N, T, emb_size/attention_heads)            
                
        if module == 'Decoder':  
            masks = torch.zeros((N, T, T), dtype=torch.bool)       # Boolean matrix. all entries are False
            # change the entries of lower triangle and diagonal to True  ??
            
            for i in range(self.attention_heads):
                XK_head = self.decoder_QKV_matrices['WK'+str(i)](X)                  # shape (N, T, emb_size/attention_heads)
                XQ_head = self.decoder_QKV_matrices['WQ'+str(i)](X)                  # shape (N, T, emb_size/attention_heads)
                similarity = torch.bmm(XK_head, XQ_head.permute(0, 2, 1))            # shape (N, T, T)
                # masking                
                similarity = similarity.masked_fill_(masks, -np.inf)                 
                alpha = F.softmax(similarity/np.sqrt(self.emb_size/self.attention_heads), dim=2)  # softmax and scaled dot product. shape (N, T, T)
                XV_head = self.decoder_QKV_matrices['WV'+str(i)](X)                  # shape (N, T, emb_size/attention_heads)
                outputs.append(torch.bmm(alpha, XV_head))                            # shape (N, T, emb_size/attention_heads)
                
        residual = torch.cat(outputs, dim=2)                  
        
        
        return residual
        
        
    def _cross_attention_step(self, Y, H):
        '''
        Perform cross-MultiHead-attention mechanism once. The key and value are from the Encoder and query is from Decoder.
        This is called by _Decoder_one_stack() and should not be called manually.
        
        Parameters
        ----------
        Y: tensor of shape (N, Ty, emb_size).
            The previous output sequence (target sentence) embeddings (the output before this MultiHead attention layer)            
        H: tensor of shape (N, Tx, emb_size)
            Encoder output vectors.
            
        Returns
        -------
        residual: tensor of shape (N, Ty, emb_size).
            Is the output of multi_head attention which can be thought as a refined word embeddings.
        
        '''
                
        # compute XK, XV, XQ (multipule heads), similarity (between XK and XQ), finally output
        outputs = []        
          
        for i in range(self.attention_heads):
            HK_head = self.decoder_QKV_matrices['WK'+str(i)](H)                  # shape (N, Tx, emb_size/attention_heads)
            YQ_head = self.decoder_QKV_matrices['WQ'+str(i)](Y)                  # shape (N, Ty, emb_size/attention_heads)
            similarity = torch.bmm(YQ_head, HK_head.permute(0, 2, 1))            # shape (N, Ty, Tx)
            alpha = F.softmax(similarity/np.sqrt(self.emb_size/self.attention_heads), dim=2)  # softmax and scaled dot product. shape (N, Ty, Tx)
            HV_head = self.decoder_QKV_matrices['WV'+str(i)](H)                  # shape (N, Tx, emb_size/attention_heads)
            outputs.append(torch.bmm(alpha, HV_head))                            # shape (N, Ty, emb_size/attention_heads)
            
        residual = torch.cat(outputs, dim=2)            
        
        
        return residual
            
            
    def _Encoder_one_stack(self, X, stack):
        ''' 
        Perform one stack of Encoder, including MultiHead attention, Residual+LayerNorm, Feed-Forward, Residual+Layernorm.
        This is called by forward() and should not be called manually.
        
        Parameters
        ----------
        X: tensor of shape (N, Tx, emb_size).
            The previous embeddings.
        stack: an interger.
            Indicates ith stack.
            
        Returns
        -------
        X: tensor of shape (N, Tx, emb_size).
            The output of Encoder_one_stack.
        '''
        
        # Multi-Head Attention
        residual = self._self_attention_step(X)     # shape (N, Tx, emb_size)
        
        # DropOut
        residual = self.dropout(residual)
        
        # Add & Norm
        X = self.layernorm(X + residual)                         # shape (N, Tx, emb_size). Keep X.
        
        # feed-forward layer
        residual = self.en_affine['Wa'+str(stack)](X)              # shape (N, Tx, emb_size)
        residual = self.relu(residual)                           # shape (N, Tx, emb_size)
        residual = self.en_affine['Wb'+str(stack)](residual)       # shape (N, Tx, emb_size)
        
        # Dropout
        residual = self.dropout(residual)
        
        # Add & Norm
        X = self.layernorm(X + residual) 
        
        return X
            
               
    def _Decoder_one_stack(self, Y, H, stack):
        ''' 
        Perform one stack of Decoder, including Masked MultiHead Self-attention, Residual+LayerNorm, 
        Multi-Head Cross-Attention, Residual+Layernorm, Feed-Forward, Residual+Layernorm.
        This is called by forward() and should not be called manually.
        
        Parameters
        ----------
        Y: tensor of shape (N, Ty, emb_size).
            The previous embeddings.
        H: tensor of shape (N, Tx, emb_size)
            Encoder output vectors.
        stack: an interger.
            Indicates ith stack.
            
        Returns
        -------
        Y: tensor of shape (N, Ty, emb_size).
            The output of Decoder_one_stack.
        '''
        
        # Masked Multi-Head Self-Attention
        residual = self._self_attention_step(Y, module = 'decoder')     # shape (N, Ty, emb_size)
        
        # DropOut
        residual = self.dropout(residual)
        
        # Add & Norm
        Y = self.layernorm(Y + residual)                         # shape (N, Ty, emb_size). Keep Y.
        
        # Multi-Head Cross-Attention
        residual = self._cross_attention_step(Y, H)              # shape (N, Ty, emb_size)
        
        # DropOut
        residual = self.dropout(residual)
        
        # Add & Norm
        Y = self.layernorm(Y + residual)                         # shape (N, Ty, emb_size). Keep Y.
        
        # feed-forward layer
        residual = self.de_affine['Wa'+str(stack)](Y)              # shape (N, Ty, emb_size)
        residual = self.relu(residual)                           # shape (N, Ty, emb_size)
        residual = self.de_affine['Wb'+str(stack)](residual)       # shape (N, Ty, emb_size)
        
        # Dropout
        residual = self.dropout(residual)
        
        # Add & Norm
        Y = self.layernorm(Y + residual) 
        
        return Y
        
        
# test
voc_size = 10
attention_heads=4
emb_size = 16
sentence_batch =torch.tensor(([1,2,8], [4,3,2]))
N, T = sentence_batch.size()
emb_input = nn.Embedding(voc_size, emb_size)
X = emb_input(sentence_batch)

model = transformer(emb_size=16, src_voc_size=10, tgt_voc_size=15, attention_heads=3, encoder_num_stack=4, decoder_num_stack=4)

src_sentence_batch = torch.tensor(([1,2,8], [4,3,2]))
tgt_sentence_batch = torch.tensor(([14, 3,9,10], [4,13,12, 6]))
output1 = model(src_sentence_batch,tgt_sentence_batch)

# go through the input embedding layer
X = self.src_emb(src_sentence_batch)    # shape (N, Tx, emb_size). X is embeddings.
Y = self.tgt_emb(tgt_sentence_batch)    # shape (N, Tx, emb_size). X is embeddings.

# go through Encoder's stacks
for stack in range(self.encoder_num_stack):
    X = self._Encoder_one_stack(X, stack)
# now X is the Encoder's output. shape (N, Tx, emb_size).

# go through Decoder's stacks
for stack in range(self.decoder_num_stack):
    Y = self._Decoder_one_stack(Y, X, stack)       # shape (N, Ty, emb_size).

# go through the last linear layer and softmax
output = self.last_affine(Y)                       # shape (N, Ty, tgt_voc_size).
output = F.softmax(output)                         # shape (N, Ty, tgt_voc_size).



params = {}
for i in range(attention_heads):
    params['WK'+str(i)] = nn.Linear(emb_size, int(emb_size/attention_heads), bias=False)   # key weight matrix
    params['WQ'+str(i)] = nn.Linear(emb_size, int(emb_size/attention_heads), bias=False)   # query weight matrix
    params['WV'+str(i)] = nn.Linear(emb_size, int(emb_size/attention_heads), bias=False)   # value weight matrix

outputs = []
for i in range(attention_heads):
    XK_head = params['WK'+str(i)](X)                  # shape (N, T, emb_size/attention_heads)
    XQ_head = params['WQ'+str(i)](X)                  # shape (N, T, emb_size/attention_heads)
    similarity = torch.bmm(XK_head, XQ_head.permute(0, 2, 1))          # shape (N, T, T)
    alpha = F.softmax(similarity, dim=2)                         # shape (N, T, T)
    XV_head = params['WV'+str(i)](X)                  # shape (N, T, emb_size/attention_heads)
    outputs.append(torch.bmm(alpha, XV_head))                            # shape (N, T, emb_size/attention_heads)
    
residual = torch.cat(outputs, dim=2)     

num_stack =2
affineW = {}
for i in range(num_stack):
    affineW['W'+str(i)] = nn.Linear(emb_size, emb_size)
            
X += residual
X = layernorm(X)           # shape (N, T, emb_size)
stack =0
residual = affineW['W'+str(stack)](X)
X += residual
X = layernorm(X) 
