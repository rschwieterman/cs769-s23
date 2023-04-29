import torch
import torch.nn.functional as F
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import torch.optim as optim
#from transformers import Speech2TextModel
#model = Speech2TextModel.from_pretrained("facebook/s2t-small-librispeech-asr")


class s2tDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.processor  = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
        # pylint: disable-next=no-member
        self.tokenizer = self.processor.tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ele = self.dataset[idx]
        return ele

    def pad_data(self, data):
  
        return data

    def collate_fn(self, pre_batch):
        #aa = time.time()
        #print("\n\nasdf")
        #self.processor.feature_extractor(pre_batch["audio"][0]["array"], sampling_rate=pre_batch["audio"][0]["sampling_rate"], return_tensors="pt")
        
        #process audio features side
        #print(type(pre_batch))
        # pylint: disable-next=no-member
        audio_in = [self.processor.feature_extractor(x["audio"]["array"], sampling_rate = x["audio"]["sampling_rate"],return_tensors="pt")["input_features"] for x in pre_batch ]
        max_audio_feature_len = max([i.shape[1] for i in audio_in])
        bs = len(audio_in)
        audio_feature_len = audio_in[0].shape[-1]
        audio_tensor = torch.zeros(bs,max_audio_feature_len,audio_feature_len)
        for i,x in enumerate(audio_in):
            x_len = x.shape[1]
            audio_tensor[i,:x_len] = x
            

        #process text token side
        
        #so far we are going with the idea of having ALL audio avail for a sample, randomly truncating the text and predicting next token
        
        sent_in = [self.tokenizer(x['text'])['input_ids'] for x in pre_batch]
        #self.tokenizer(pre_batch['text'],padding=True)

        padding_id = 1

        cutoffs = [np.random.randint(0,len(x)-1) for x in sent_in]
        input_sents = []
        label_sents=[]
        for i,x in enumerate(sent_in):
            input_sents.append(x[:cutoffs[i]])
            label_sents.append(x[1:cutoffs[i]+1])
        #print(pre_batch)
        max_sent_len = max([len(x) for x in input_sents])
        
        sent_tokens_in_tensor = torch.zeros(bs,max_sent_len) + padding_id
        label_sents_tensor = torch.zeros(bs,max_sent_len)+ padding_id

        mask_in_tensor  =torch.zeros(bs,max_sent_len )
        for i,x in enumerate(input_sents):
            sent_tokens_in_tensor[i,:len(x)] = torch.tensor(x)
            label_sents_tensor[i,:len(x)] = torch.tensor(label_sents[i])
            mask_in_tensor[i,:len(x)] = 1

        return audio_tensor, sent_tokens_in_tensor.long(), mask_in_tensor, label_sents_tensor.long()


from s2tmodel import customs2t
model = customs2t()
#from transformers import Speech2TextModel
#model = Speech2TextModel.from_pretrained("facebook/s2t-small-librispeech-asr")
#catter = Speech2TextModel().from_pretrained("facebook/s2t-small-librispeech-asr")
old_model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

print("here")

processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
# pylint: disable-next=no-member
feature_extractor = processor.feature_extractor 
#ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds_train =load_dataset("librispeech_asr", "clean", split="train.100")
ds_validate =load_dataset("librispeech_asr", "clean", split="validation")



s2tData = s2tDataset( ds_train, None)
valData = s2tDataset( ds_validate, None)

#tester  = ds2[:10]
#test_data = s2tData.collate_fn(tester)
#outter2 = model.forward(test_data[0], decoder_input_ids=test_data[1], decoder_attention_mask=test_data[2])

train_dataloader = DataLoader(s2tData, shuffle=True, batch_size=24,collate_fn=s2tData.collate_fn)
val_dataloader = DataLoader(valData, shuffle=False, batch_size=24,collate_fn=s2tData.collate_fn)

def loss_func(pred,label,mask):
    #pdb.set_trace()
    logits = F.log_softmax(pred,dim=-1)
    bs = pred.shape[0]
    t_losses = torch.zeros(bs)
    for b in range(bs):
        t_losses[b] = torch.sum(mask[b] * F.cross_entropy(logits[b],label[b], reduction='none'))
    return t_losses.sum()/bs


optimizer = optim.AdamW(model.parameters(), lr=1e-5)
print("begin training")

for epoch in range(10):
    epoch_loss = 0.0
    v_epoch_loss = 0.0
    for batch_data in tqdm(train_dataloader):
        optimizer.zero_grad()
        outter2 = model.forward(batch_data[0].to(device), decoder_input_ids=batch_data[1].to(device), decoder_attention_mask=batch_data[2].to(device))
        loss = loss_func(outter2.cls_logit_out, batch_data[3].to(device), batch_data[2].to(device))
        loss.backward()
        epoch_loss += loss.detach().cpu().item()
        optimizer.step()
    for val_data in tqdm(val_dataloader):
        optimizer.zero_grad()
        #print("i need to know about data")
        voutputs = model.forward(val_data[0].to(device).detach(), decoder_input_ids=val_data[1].to(device).detach(), decoder_attention_mask=val_data[2].to(device).detach())
        #voutputs = voutputs.detach()
        vloss = loss_func(voutputs.cls_logit_out, val_data[3].to(device), val_data[2].to(device))
        v_epoch_loss += vloss
    print(epoch_loss)
    print(epoch_loss/len(train_dataloader))
    print(v_epoch_loss/len(val_dataloader))



exit()

inputs = feature_extractor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
input_features = inputs.input_features
decoder_input_ids_in = torch.tensor([[1, 1]]) * model.transformer.config.decoder_start_token_id
generated_ids = old_model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
old_hidden = old_model(input_features, decoder_input_ids=decoder_input_ids_in)
outter = model.forward(input_features, decoder_input_ids=decoder_input_ids_in)





last_hidden_state = model.forward(input_features, decoder_input_ids=decoder_input_ids_in).last_hidden_state
print(last_hidden_state)
#encode is ours

exit()
model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")




generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(transcription)
print("done")
