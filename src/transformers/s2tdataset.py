import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Speech2TextProcessor


class S2TDataset(Dataset):
    default_keywords = ["up", "down", "left", "right", "yes", "no", "on", "off", "go", "stop"]

    def __init__(self, dataset, args, keywords=None):
        self.dataset = dataset
        self.p = args
        self.processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
        # pylint: disable-next=no-member
        self.tokenizer = self.processor.tokenizer

        if keywords is None:
            keywords = S2TDataset.default_keywords
        self.keyword_ids = np.array(list(self.tokenizer(keyword)["input_ids"][0] for keyword in keywords))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ele = self.dataset[idx]
        return ele

    def pad_data(self, data):
        return data

    def collate_fn(self, pre_batch):
        # aa = time.time()
        # print("\n\nasdf")
        # self.processor.feature_extractor(pre_batch["audio"][0]["array"], sampling_rate=pre_batch["audio"][0]["sampling_rate"], return_tensors="pt")

        # process audio features side
        # print(type(pre_batch))
        audio_in = [
            # pylint: disable-next=no-member
            self.processor.feature_extractor(
                x["audio"]["array"], sampling_rate=x["audio"]["sampling_rate"], return_tensors="pt"
            )["input_features"]
            for x in pre_batch
        ]
        max_audio_feature_len = max([i.shape[1] for i in audio_in])
        bs = len(audio_in)
        audio_feature_len = audio_in[0].shape[-1]
        audio_tensor = torch.zeros(bs, max_audio_feature_len, audio_feature_len)
        for i, x in enumerate(audio_in):
            x_len = x.shape[1]
            audio_tensor[i, :x_len] = x

        # process text token side

        # so far we are going with the idea of having ALL audio avail for a sample, randomly truncating the text and predicting next token

        sent_in = [self.tokenizer(x["text"])["input_ids"] for x in pre_batch]
        # self.tokenizer(pre_batch['text'],padding=True)

        keywords_tensor = np.array(list(list(keyword_id in sent for keyword_id in self.keyword_ids) for sent in sent_in), dtype=int)

        padding_id = self.tokenizer(text="<pad>")["input_ids"][0]

        cutoffs = [np.random.randint(0, len(x) - 1) for x in sent_in]
        input_sents = []
        label_sents = []
        for i, x in enumerate(sent_in):
            input_sents.append(x[: cutoffs[i]])
            label_sents.append(x[1 : cutoffs[i] + 1])
        # print(pre_batch)
        max_sent_len = max([len(x) for x in input_sents])

        sent_tokens_in_tensor = torch.zeros(bs, max_sent_len) + padding_id
        label_sents_tensor = torch.zeros(bs, max_sent_len) + padding_id

        mask_in_tensor = torch.zeros(bs, max_sent_len)
        for i, x in enumerate(input_sents):
            sent_tokens_in_tensor[i, : len(x)] = torch.tensor(x)
            label_sents_tensor[i, : len(x)] = torch.tensor(label_sents[i])
            mask_in_tensor[i, : len(x)] = 1

        return (
            audio_tensor,
            sent_tokens_in_tensor.long(),
            mask_in_tensor,
            label_sents_tensor.long(),
            keywords_tensor,
        )
