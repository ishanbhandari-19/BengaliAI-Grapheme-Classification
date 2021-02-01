from tqdm import tqdm
import torch

from utils import macro_recall, loss_fn


def train(dataset, data_loader, model, optimizer):
    model.train()
    final_loss = 0
    counter = 0
    final_outputs = []
    final_targets = []

    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):
        counter = counter + 1
        image = d["image"]
        grapheme_root = d["grapheme_root"]
        vowel_diacritic = d["vowel_diacritic"]
        consonant_diacritic = d["consonant_diacritic"]
        DEVICE = "cuda"
        image = image.to(DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)
        


        optimizer.zero_grad()
        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        final_loss += loss

        o1, o2, o3 = outputs
        t1, t2, t3 = targets
        final_outputs.append(torch.cat((o1,o2,o3), dim=1))
        final_targets.append(torch.stack((t1,t2,t3), dim=1))

        #if bi % 10 == 0:
        #    break
    final_outputs = torch.cat(final_outputs)
    final_targets = torch.cat(final_targets)

    print("=================Train=================")
    macro_recall_score = macro_recall(final_outputs, final_targets)
    
    return final_loss/counter , macro_recall_score



def evaluate(dataset, data_loader, model):
    with torch.no_grad():
        model.eval()
        final_loss = 0
        counter = 0
        final_outputs = []
        final_targets = []
        for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):
            counter = counter + 1
            image = d["image"]
            grapheme_root = d["grapheme_root"]
            vowel_diacritic = d["vowel_diacritic"]
            consonant_diacritic = d["consonant_diacritic"]
            DEVICE = "cuda"
            image = image.to(DEVICE, dtype=torch.float)
            grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
            vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
            consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

            outputs = model(image)
            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
            loss = loss_fn(outputs, targets)
            final_loss += loss

            o1, o2, o3 = outputs
            t1, t2, t3 = targets
            #print(t1.shape)
            final_outputs.append(torch.cat((o1,o2,o3), dim=1))
            final_targets.append(torch.stack((t1,t2,t3), dim=1))
        
        final_outputs = torch.cat(final_outputs)
        final_targets = torch.cat(final_targets)

        print("=================VALID============")
        macro_recall_score = macro_recall(final_outputs, final_targets)

    return final_loss/counter , macro_recall_score

