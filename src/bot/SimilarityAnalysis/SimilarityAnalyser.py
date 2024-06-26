from dataclasses import dataclass
from typing import Optional, Any, Annotated, Union, Callable

import torch
from torch import tensor, float32


@dataclass
class ValueRange:
    min_value: float
    max_value: float

    def validate(self, x: float) -> float:
        if not (self.min_value <= x <= self.max_value):
            raise ValueError(f"{x} must be in range [{self.min_value}, {self.max_value}]")
        return x


class SimilarityAnalyser:
    def __init__(self, threshold: Annotated[float, ValueRange(0, 1)], dtype: Optional[Any] = float32,
                 device: Optional[str] = "cpu") -> None:
        self.__threshold: float = ValueRange(0, 1).validate(threshold)
        self.__dtype: Any = dtype
        self.__device: str = device
        self.__cached_representations: tensor = None
        self.__cached_ideal_representation: tensor = None
        self.__cached_similarity: tensor = None

    @property
    def threshold(self) -> float:
        return self.__threshold

    @threshold.setter
    def threshold(self, new_threshold: Annotated[float, ValueRange(0, 1)]) -> tensor:
        self.__threshold = ValueRange(0, 1).validate(new_threshold)
        self.__call__(self.__cached_representations, self.__cached_ideal_representation)

    def __calculate_ratio_of_length_of_representation(self) -> tensor:
        length_of_representations: tensor = torch.norm(self.__cached_representations, dim=1)
        length_of_ideal_representation: tensor = torch.norm(self.__cached_ideal_representation)

        return length_of_representations / length_of_ideal_representation

    def __call__(
            self,
            representations: Union[list[tensor], tensor],
            ideal_representation: tensor
    ) -> tensor:

        self.__cached_representations = (
            representations if type(representations) is tensor else torch.stack(representations))
        self.__cached_representations = (
            self.__cached_representations.clone().detach().to(dtype=self.__dtype, device=self.__device))

        self.__cached_ideal_representation = (
            ideal_representation.clone().detach().to(dtype=self.__dtype, device=self.__device))

        cosine_similarity: tensor = torch.cosine_similarity(self.__cached_representations,
                                                            self.__cached_ideal_representation)
        ratio_of_representations: tensor = self.__calculate_ratio_of_length_of_representation()
        self.__cached_similarity = torch.clamp(cosine_similarity * ratio_of_representations, 0, 1)

        return self.__cached_similarity

    @staticmethod
    def __get_indices_of_filtered_tensor(target_tensor: tensor,
                                         filter_func: Callable[[tensor], tensor]
                                         ) -> tensor:
        mask: tensor = filter_func(target_tensor)

        return torch.nonzero(mask)

    def get_max_similarity(self) -> float:
        valid_similarity_indices: tensor = self.__get_indices_of_filtered_tensor(self.__cached_similarity,
                                                                                 lambda x: x <= self.__threshold)
        if len(valid_similarity_indices) == 0:
            return 0

        valid_similarity: tensor = self.__cached_similarity[valid_similarity_indices]
        return torch.max(valid_similarity).unique().item()

    def get_representation_with_max_similarity(self, max_similarity: float) -> list:
        representation_with_max_similarity_indices: tensor = (
            self.__get_indices_of_filtered_tensor(self.__cached_similarity, lambda x: x == max_similarity))

        return self.__cached_representations[representation_with_max_similarity_indices].squeeze().tolist()

    def get_most_similar_representation(self) -> dict[tensor, float]:
        max_similarity: float = self.get_max_similarity()
        representations_with_max_similarity: list = self.get_representation_with_max_similarity(max_similarity)

        return {"representations": representations_with_max_similarity, "similarity": max_similarity}

    def get_most_similar_representation_index(self) -> int:
        max_similarity_value: float = self.get_most_similar_representation()["similarity"]

        return (self.__get_indices_of_filtered_tensor(self.__cached_similarity, lambda x: x == max_similarity_value)
                .squeeze()
                .tolist())
