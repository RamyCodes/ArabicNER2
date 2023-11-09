import os
import logging
import torch
import numpy as np
from arabiner.trainers import BaseTrainer
from arabiner.utils.metrics import compute_single_label_metrics

logger = logging.getLogger(__name__)


class BertTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self):
        test_loss = np.inf
        num_train_batch = len(self.train_dataloader)
        patience = self.patience

        for epoch_index in range(self.max_epochs):
            self.current_epoch = epoch_index
            train_loss = 0

            for batch_index, (_, gold_tags, _, _, logits) in enumerate(self.tag(
                self.train_dataloader, is_train=True
            ), 1):
                self.current_timestep += 1
                batch_loss = self.loss(logits.view(-1, logits.shape[-1]), gold_tags.view(-1))
                batch_loss.backward()

                # Avoid exploding gradient by doing gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                self.optimizer.step()
                self.scheduler.step()
                train_loss += batch_loss.item()

                if self.current_timestep % self.log_interval == 0:
                    logger.info(
                        "Epoch %d | Batch %d/%d | Timestep %d | LR %.10f | Loss %f",
                        epoch_index,
                        batch_index,
                        num_train_batch,
                        self.current_timestep,
                        self.optimizer.param_groups[0]['lr'],
                        batch_loss.item()
                    )

            train_loss /= num_train_batch

            logger.info("** Evaluating on validation dataset **")

            epoch_summary_loss = {
                "train_loss": train_loss,
                "val_loss": train_loss
            }
            epoch_summary_metrics = {
                "val_micro_f1": train_loss,
                "val_precision": train_loss,
                "val_recall": train_loss
            }

            logger.info(
                "Epoch %d | Timestep %d | Train Loss %f | Val Loss %f | F1 %f",
                epoch_index,
                self.current_timestep,
                train_loss,
            )

            patience = self.patience
            logger.info("** Validation improved, evaluating test data **")
            test_preds, segments, valid_len, test_loss = self.eval(self.test_dataloader)
            self.segments_to_file(segments, os.path.join(self.output_path, "predictions.txt"))
            test_metrics = compute_single_label_metrics(segments)

            epoch_summary_loss["test_loss"] = test_loss
            epoch_summary_metrics["test_micro_f1"] = test_metrics.micro_f1
            epoch_summary_metrics["test_precision"] = test_metrics.precision
            epoch_summary_metrics["test_recall"] = test_metrics.recall

            logger.info(
                f"Epoch %d | Timestep %d | Test Loss %f | F1 %f",
                epoch_index,
                self.current_timestep,
                test_loss,
                test_metrics.micro_f1
            )

            self.save()

            # No improvements, terminating early
            if patience == 0:
                logger.info("Early termination triggered")
                break

            self.summary_writer.add_scalars("Loss", epoch_summary_loss, global_step=self.current_timestep)
            self.summary_writer.add_scalars("Metrics", epoch_summary_metrics, global_step=self.current_timestep)

    def eval(self, dataloader):
        golds, preds, segments, valid_lens = list(), list(), list(), list()
        loss = 0

        for _, gold_tags, tokens, valid_len, logits in self.tag(
            dataloader, is_train=False
        ):
            loss += self.loss(logits.view(-1, logits.shape[-1]), gold_tags.view(-1))
            preds += torch.argmax(logits, dim=2).detach().cpu().numpy().tolist()
            segments += tokens
            valid_lens += list(valid_len)

        loss /= len(dataloader)

        # Update segments, attach predicted tags to each token
        segments = self.to_segments(segments, preds, valid_lens, dataloader.dataset.vocab)

        return preds, segments, valid_lens, loss.item()

    def infer(self, dataloader):
        golds, preds, segments, valid_lens = list(), list(), list(), list()

        for _, gold_tags, tokens, valid_len, logits in self.tag(
            dataloader, is_train=False
        ):
            preds += torch.argmax(logits, dim=2).detach().cpu().numpy().tolist()
            segments += tokens
            valid_lens += list(valid_len)

        segments = self.to_segments(segments, preds, valid_lens, dataloader.dataset.vocab)
        return segments

    def to_segments(self, segments, preds, valid_lens, vocab):
        if vocab is None:
            vocab = self.vocab

        tagged_segments = list()
        tokens_stoi = vocab.tokens.get_stoi()
        tags_itos = vocab.tags[0].get_itos()
        unk_id = tokens_stoi["UNK"]

        for segment, pred, valid_len in zip(segments, preds, valid_lens):
            # First, the token at 0th index [CLS] and token at nth index [SEP]
            # Combine the tokens with their corresponding predictions
            segment_pred = zip(segment[1:valid_len-1], pred[1:valid_len-1])

            # Ignore the sub-tokens/subwords, which are identified with text being UNK
            segment_pred = list(filter(lambda t: tokens_stoi[t[0].text] != unk_id, segment_pred))

            # Attach the predicted tags to each token
            list(map(lambda t: setattr(t[0], 'pred_tag', [{"tag": tags_itos[t[1]]}]), segment_pred))

            # We are only interested in the tagged tokens, we do no longer need raw model predictions
            tagged_segment = [t for t, _ in segment_pred]
            tagged_segments.append(tagged_segment)

        return tagged_segments
