import torch
import torch.nn.functional as nn
from lightning import LightningModule
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy

from .Attention import *


class EmotionModel(LightningModule):
    def __init__(self,
                 attention: str,
                 dropout: Optional[float] = 0.5,
                 scaler: Optional[float] = None,
                 bias: Optional[bool] = True,
                 dtype: Optional[Any] = torch.float32,
                 device: Optional[str] = "cpu") -> None:
        super(EmotionModel, self).__init__()

        self.__dtype: Any = dtype
        self.__device: str = device

        match attention:
            case "dot_product":
                self.__attention = DotProductAttention()
            case "scaled_dot_product":
                self.__attention = ScaledDotProductAttention(scaler)
            case "additive":
                self.__attention = AdditiveAttention(dropout, dtype=dtype, device=device)
            case "dual_linear":
                self.__attention = DualLinearAttention(dropout, dtype=dtype, device=device)

        self.__weight = torch.nn.LazyLinear(7, bias=bias, device=device, dtype=dtype)

        self.__train_prediction: list = []
        self.__validation_prediction: list = []
        self.__evaluation_prediction: list = []


    def forward(self, representation: torch.tensor, input_emotion: torch.tensor) -> torch.tensor:
        decomposed_representation: torch.tensor = representation.diag()

        decomposed_attention_matrix: torch.tensor = self.__attention(input_emotion, decomposed_representation)
        decomposed_attention_matrix = decomposed_attention_matrix.to(dtype=self.__dtype, device=self.__device)

        attention_matrix: torch.tensor = torch.sum(decomposed_attention_matrix, dim=1, dtype=self.__dtype)

        attention_score: torch.tensor = torch.softmax(attention_matrix, dim=0, dtype=self.__dtype)

        new_representation: torch.tensor = torch.clamp(self.__weight(attention_score), -1, 1)

        return new_representation

    def representation_evolution(self, representation_src: list, emotion_compositions: list) -> list:
        representation: list = representation_src
        for composition in emotion_compositions:
            new_representation: torch.tensor = self.forward(torch.tensor(representation[-1], dtype=self.__dtype, device=self.__device),
                                                            torch.tensor(composition, dtype=self.__dtype, device=self.__device))
            representation.append(new_representation)

        return representation

    def training_step(self, batch, batch_idx) -> float:
        data, label = batch

        output: list = self.representation_evolution(data[0], data[1])
        prediction: list = [torch.argmax(representation) for representation in output]
        self.__train_prediction.append({
            "prediction": torch.tensor(prediction),
            "truth": torch.tensor(label)
        })

        loss = nn.cross_entropy(torch.tensor(prediction), torch.tensor(label))

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_train_epoch_end(self) -> tuple:
        all_prediction: torch.tensor = torch.cat([turn["prediction"] for turn in self.__train_prediction])
        all_truth: torch.tensor = torch.cat([turn["truth"] for turn in self.__train_prediction])

        f1_score = multiclass_f1_score(all_truth, all_prediction, num_classes=7, average="micro")
        accuracy = multiclass_accuracy(all_truth, all_prediction, num_classes=7)

        self.log("f1_score", f1_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return f1_score, accuracy

    def on_validation_epoch_start(self) -> None:
        self.__validation_prediction = []

    def validation_step(self, batch, batch_idx) -> float:
        data, label = batch

        output: list = self.representation_evolution(data[0], data[1])
        prediction: list = [torch.argmax(representation) for representation in output]
        self.__validation_prediction.append({
            "prediction": torch.tensor(prediction),
            "truth": torch.tensor(label)
        })

        loss = nn.cross_entropy(torch.tensor(prediction), torch.tensor(label))

        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_validation_epoch_end(self) -> tuple:
        all_prediction: torch.tensor = torch.cat([turn["prediction"] for turn in self.__train_prediction])
        all_truth: torch.tensor = torch.cat([turn["truth"] for turn in self.__train_prediction])

        f1_score = multiclass_f1_score(all_truth, all_prediction, num_classes=7, average="micro")
        accuracy = multiclass_accuracy(all_truth, all_prediction, num_classes=7)

        self.log("f1_score", f1_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return f1_score, accuracy