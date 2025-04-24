from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, TypeVar, Union

from sm.evaluation.prelude import PrecisionRecallF1Support
from sm.evaluation.utils import PrecisionRecallF1
from sm.misc.matrix import Matrix


@dataclass
class ConfusionMatrix:
    tp: int = 0  # true positive
    fn: int = 0  # false negative
    fp: int = 0  # false positive
    tn: int = 0  # true negative

    def total(self):
        return self.tp + self.fn + self.fp + self.tn

    def precision(self):
        if self.total() == 0:
            return 1.0
        # the system must always predict something. NO_ENTITY is treated as a prediction
        if self.tp + self.fp > 0:
            return self.tp / (self.tp + self.fp)
        return 0.0

    def recall(self):
        if self.total() == 0:
            return 1.0
        # the system must always predict something. NO_ENTITY is treated as a prediction
        if self.tp + self.fn > 0:
            return self.tp / (self.tp + self.fn)
        return 0.0

    def f1(self):
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def get_precision_recall_f1(self) -> PrecisionRecallF1:
        return PrecisionRecallF1(self.precision(), self.recall(), self.f1())

    def __add__(self, another: ConfusionMatrix):
        return ConfusionMatrix(
            self.tp + another.tp,
            self.fn + another.fn,
            self.fp + another.fp,
            self.tn + another.tn,
        )


def inkb_eval_table(
    gold_ents: Matrix[set[str]],
    pred_ents: Matrix[list[str]],
    k: Optional[Union[int, Sequence[Optional[int]]]] = None,
) -> tuple[dict[str, PrecisionRecallF1Support], dict[str, ConfusionMatrix]]:
    """Evaluate entity linking performance on a table using InKB metrics. As defined in Gerbil framework, InKB metrics
    only consider entities in KB as correct entities, no entities (incorrect mentions) or entities outside of KB (NIL) are
    consider incorrect and eliminated. In other words, only mentions with valid KB entities are used for evaluation.

    NOTE: This evaluation assumes that no pred entity is still a prediction (NO_ENTITY). Therefore, the system will never be able to
    achieve 1.0 precision by not predicting anything.

    Args:
        gold_ents: Matrix of gold entities for each cell. If the set of entities of a cell is empty, the
            cell does not contain any entity (no entity in KB).
        pred_ents: Matrix of predicted entities for each cell. If the list of candidate entities of a
            cell is empty, we do not predict any entity for the cell. Candidates in the list are sorted by
            their likelihood in descending order.
    """
    nrows, ncols = gold_ents.shape()
    name2k: dict[str, Optional[int]] = {"perf@all": None}
    if k is not None:
        for ki in k if isinstance(k, Sequence) else [k]:
            if ki is not None:
                name2k[f"perf@{ki}"] = ki

    confusion_matrices = {name: ConfusionMatrix() for name in name2k}

    for i in range(nrows):
        for j in range(ncols):
            if len(gold_ents[i, j]) == 0:
                # no entity in gold, but we predict something, this is obviously wrong
                # but since we ignore no entity or incorrect mentions, we do not count
                # this as a false positive
                continue

                # if len(pred_ents[i, j]) == 0:
                #     # no entity in gold, no entity in pred, so no example
                #     continue
                # for name in name2k:
                #     confusion_matrices[name].fp += 1
            else:
                ytrue = gold_ents[i, j]
                ypreds = pred_ents[i, j]

                for name, k in name2k.items():
                    if len(ytrue.intersection(ypreds[:k])) > 0:
                        confusion_matrices[name].tp += 1
                    else:
                        confusion_matrices[name].fn += 1
                        # so recall = precision
                        confusion_matrices[name].fp += 1

    output = {}
    for name, ki in name2k.items():
        cm = confusion_matrices[name]
        output[name] = PrecisionRecallF1Support(
            recall=cm.recall(),
            precision=cm.precision(),
            f1=cm.f1(),
            support=cm.tp + cm.fn,
        )

    return output, confusion_matrices


T = TypeVar("T", str, int, tuple[str, ...])


def inkb_mrr(
    ytrue: Sequence[T] | Sequence[set[T]] | Matrix[set[T]],
    ypreds: Sequence[Sequence[T]] | Matrix[Sequence[T]],
    is_in_groundtruth: Optional[Callable[[T, set[T]], bool]] = None,
) -> float:
    """Calculate MRR for a list of queries. Optionally support passing a matrix.

    This follow the GERBIL InKB metrics. Only mentions with valid KB entities are used for evaluation.
    """
    if is_in_groundtruth is None:
        is_in_groundtruth = lambda ypred, ytrues: ypred in ytrues

    if isinstance(ytrue, Matrix):
        assert isinstance(ypreds, Matrix)
        nrows, ncols = ytrue.shape()
        out = 0.0
        n = 0
        for ri in range(nrows):
            for ci in range(ncols):
                y = ytrue[ri, ci]
                if len(y) == 0:
                    # skip mentions with no entity
                    continue

                out += next(
                    (
                        1 / j
                        for j, ypred in enumerate(ypreds[ri, ci], start=1)
                        if is_in_groundtruth(ypred, y)
                    ),
                    0,
                )
                n += 1

        return out / max(n, 1)

    if len(ytrue) == 0:
        return 0.0

    out = 0.0

    assert isinstance(ypreds, Sequence)

    if isinstance(ytrue[0], str):
        for i in range(len(ytrue)):
            y = ytrue[i]
            out += next(
                (
                    1 / j
                    for j, ypred in enumerate(ypreds[i], start=1)
                    if is_in_groundtruth(ypred, {y})  # type: ignore
                ),
                0,
            )
    else:
        for i in range(len(ytrue)):
            y = ytrue[i]
            assert len(y) > 0  # type: ignore
            out += next(
                (
                    1 / j
                    for j, ypred in enumerate(ypreds[i], start=1)
                    if is_in_groundtruth(ypred, y)  # type: ignore
                ),
                0,
            )
    return out / max(len(ytrue), 1)
