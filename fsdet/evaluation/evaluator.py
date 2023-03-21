import datetime
import json
import logging
import os
import time
from collections import OrderedDict, abc
from contextlib import contextmanager, ExitStack
from torch import nn
import torch
import detectron2
from detectron2.utils.comm import is_main_process
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import log_every_n_seconds


def rec_intersection(bbox1, bbox2):
    dx = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
    dy = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
    if (dx > 0) and (dy > 0):
        return dx * dy
    else:
        return 0


def rec_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(
                        k
                    )
                    results[k] = v
        return results


def inference_on_dataset(model, data_loader, evaluator, dataset_name):
    # FIXME[DONE]: called in evaluation stage of training/fine-tuning
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    metadata = MetadataCatalog.get(dataset_name)
    category_name = dataset_name.split("mvtecvoc_test_all_")[-1]

    file_split = "test"  # FIXME: train, validation, or test
    anomaly_type = (
        "good" if file_split in ["validation", "train"] else "structural_anomalies"
    )  # FIXME:  good, structural_anomalies, logical_anomalies

    output_dir_name = f"{file_split}_{anomaly_type}_{category_name}"
    os.makedirs(output_dir_name, exist_ok=True)

    prediction_output = []

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            # FIXME[DONE]: update code here to give visual outputs
            for input, output in zip(inputs, outputs):
                pred_out = {
                    "category": category_name,
                    "file_split": file_split,
                    "anomaly_type": anomaly_type,
                    "image_id": input["image_id"],
                    "height": input["height"],
                    "width": input["width"],
                }
                image = read_image(input["file_name"], format="BGR")
                image = image[:, :, ::-1]

                visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
                if "instances" in output:
                    instances = output["instances"].to(torch.device("cpu"))

                    novel_classes = (
                        ["pushpin"]
                        if category_name == "pushpins"
                        else metadata.get("novel_classes")
                    )
                    all_classes = metadata.get("thing_classes")
                    novel_classes_ordinal = [
                        all_classes.index(c) for c in novel_classes
                    ]

                    # The following three variables are all tensors
                    pred_classes = instances.pred_classes
                    scores = instances.scores
                    boxes = instances.pred_boxes.tensor

                    """
                    Option 1: choose top-k
                    """
                    # final_preds = []

                    # for c in novel_classes:
                    #     c_ordinal = all_classes.index(c)
                    #     c_pred_idx = torch.nonzero(pred_classes == c_ordinal).squeeze()
                    #     c_pred_scores = torch.index_select(scores, 0, c_pred_idx)

                    #     # FIXME[DONE]: other categories
                    #     if c == "orange":
                    #         topk = 2
                    #     else:
                    #         topk = 1

                    #     (values, indices) = torch.topk(
                    #         c_pred_scores,
                    #         k=(
                    #             c_pred_idx.shape[0]
                    #             if topk > c_pred_idx.shape[0]
                    #             else topk
                    #         ),
                    #         dim=0,
                    #     )
                    #     final_preds_4c = torch.index_select(c_pred_idx, 0, indices)
                    #     final_preds.append(final_preds_4c)

                    # final_preds = torch.cat(final_preds, dim=0)

                    """
                    Option 2: filter by score
                    """

                    candidate_preds = torch.nonzero(
                        sum(pred_classes == i for i in novel_classes_ordinal)
                        & (scores > 0.2)  # FIXME[DONE]: score confidence threshold
                    ).squeeze()  # a tensor of indices of plausible predictions

                    # high IoU filtering
                    cand_preds_count = candidate_preds.shape[0]
                    remove_indices = set()
                    iou_threshold = 0.5  # FIXME[DONE]: choose threshold

                    for i in range(cand_preds_count - 1):
                        for j in range(i + 1, cand_preds_count):
                            if (i in remove_indices) or (j in remove_indices):
                                continue
                            score_i = scores[candidate_preds[i]]
                            score_j = scores[candidate_preds[j]]
                            box_i = boxes[candidate_preds[i]]
                            box_j = boxes[candidate_preds[j]]

                            intersection_area = rec_intersection(box_i, box_j)
                            union_area = (
                                rec_area(box_i) + rec_area(box_j) - intersection_area
                            )

                            if intersection_area / union_area > iou_threshold:
                                remove_indices.add(j if score_i >= score_j else i)

                    keep_preds = set(range(cand_preds_count)) - remove_indices
                    final_preds = torch.index_select(
                        candidate_preds, 0, torch.tensor(list(keep_preds))
                    )

                    scores = torch.index_select(scores, 0, final_preds)
                    pred_classes = torch.index_select(pred_classes, 0, final_preds)
                    boxes = torch.index_select(boxes, 0, final_preds)

                    # update pred_out
                    pred_out["scores"] = scores.detach().tolist()
                    pred_out["pred_classes"] = [
                        all_classes[i] for i in pred_classes.detach().tolist()
                    ]
                    pred_out["boxes"] = boxes.detach().tolist()

                    novel_instances = detectron2.structures.Instances(
                        image_size=instances.image_size
                    )

                    novel_instances.set("pred_classes", pred_classes)
                    novel_instances.set("scores", scores)
                    novel_instances.set(
                        "pred_boxes", detectron2.structures.Boxes(tensor=boxes)
                    )

                    vis_output = visualizer.draw_instance_predictions(
                        predictions=novel_instances
                    )

                vis_output.save(
                    os.path.join(output_dir_name, os.path.basename(input["file_name"]),)
                )

                prediction_output.append(pred_out)

            with open(
                os.path.join(output_dir_name, f"{output_dir_name}.json"), "w"
            ) as f:
                json.dump(prediction_output, f)

            start_eval_time = time.perf_counter()
            # evaluator.process(inputs, outputs) #FIXME[DONE]: uncomment me for proper evaluation
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (
                time.perf_counter() - start_time
            ) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_iter * (total - idx - 1))
                )
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices,
        )
    )

    # results = evaluator.evaluate()  #FIXME[DONE]: uncomment me for proper evaluation
    results = OrderedDict()
    results["bbox"] = {
        "AP": 0.5,
        "AP50": 0.5,
        "AP75": 0.5,
    }
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
