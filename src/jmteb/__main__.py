from __future__ import annotations

import json
from pathlib import Path

from jsonargparse import ActionConfigFile, ArgumentParser
from loguru import logger

from jmteb.embedders import TextEmbedder
from jmteb.evaluators import EmbeddingEvaluator
from jmteb.utils.score_recorder import JsonScoreRecorder


def main(
    text_embedder: TextEmbedder,
    evaluators: dict[str, EmbeddingEvaluator],
    save_dir: str | None = None,
    overwrite_cache: bool = False,
    prompt_templates: str | None = None,
    model_name_or_path: str | None = None
):
    logger.info(f"Start evaluating the following tasks\n{list(evaluators.keys())}")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    score_recorder = JsonScoreRecorder(save_dir)

    if prompt_templates:
        with open(prompt_templates) as f:
            prompt_templates = json.load(f)

    print("--------------------")
    print(prompt_templates)
    print("--------------------")

    for eval_name, evaluator in evaluators.items():
        logger.info(f"Evaluating {eval_name}")

        cache_dir = None
        if save_dir is not None:
            cache_dir = Path(save_dir) / "cache" / eval_name

        if prompt_templates:
            query_template = ""
            passage_template = ""

            model_id = text_embedder.model_name_or_path
            task = eval_name
            print(f"model_id:{model_id}, task:{task}")
            if model_id in prompt_templates:
                if "query" in prompt_templates[model_id]:
                    if type(prompt_templates[model_id]["query"]) == str:
                        query_template = prompt_templates[model_id]["query"]
                    else:
                        if task in prompt_templates[model_id]["query"]:
                            query_template = prompt_templates[model_id]["query"][task]
                if "passage" in prompt_templates[model_id]:
                    if type(prompt_templates[model_id]["passage"]) == str:
                        passage_template = prompt_templates[model_id]["passage"]
                    else:
                        if task in prompt_templates[model_id]["passage"]:
                            passage_template = prompt_templates[model_id]["passage"][task]
                evaluator.query_template = query_template
                evaluator.passage_template = passage_template

        print(f"query:{query_template}, passage:{passage_template}")
        metrics = evaluator(text_embedder, cache_dir=cache_dir, overwrite_cache=overwrite_cache)


        score_recorder.record_task_scores(
            scores=metrics,
            dataset_name=eval_name,
            task_name=evaluator.__class__.__name__.replace("Evaluator", ""),
        )

        

        logger.info(f"Results for {eval_name}\n{json.dumps(metrics.as_dict(), indent=4, ensure_ascii=False)}")

    logger.info(f"Saving result summary to {Path(save_dir) / 'summary.json'}")
    score_recorder.record_summary()


if __name__ == "__main__":
    parser = ArgumentParser(parser_mode="jsonnet")

    parser.add_subclass_arguments(TextEmbedder, nested_key="embedder", required=True)
    parser.add_argument(
        "--evaluators",
        type=dict[str, EmbeddingEvaluator],
        enable_path=True,
        default=str(Path(__file__).parent / "configs" / "jmteb.jsonnet"),
    )
    parser.add_argument("--config", action=ActionConfigFile, help="Path to the config file.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the outputs")
    parser.add_argument("--overwrite_cache", type=bool, default=False, help="Overwrite the save_dir if it exists")
    parser.add_argument("--eval_exclude", type=list[str], default=None, help="Evaluators to exclude.")
    parser.add_argument("--prompt_templates", type=str, default=str(Path(__file__).parent / "configs" / "prompt_templates.json"), help="Path to the prefix file")

    args = parser.parse_args()

    if args.eval_exclude is not None:
        evaluator_keys = list(args.evaluators.keys())
        # remove evaluators in eval_exclude
        for key in evaluator_keys:
            if key in args.eval_exclude:
                args.evaluators.pop(key)

    args = parser.instantiate_classes(args)
    if isinstance(args.evaluators, str):
        raise ValueError(
            "Evaluators should be a dictionary, not a string.\n"
            "Perhaps you provided a path to a config file, "
            "but the path does not exist or the config format is broken.\n"
            f"Please check {args.evaluators}"
        )

    main(
        text_embedder=args.embedder,
        evaluators=args.evaluators,
        save_dir=args.save_dir,
        overwrite_cache=args.overwrite_cache,
        prompt_templates=args.prompt_templates,
    )
