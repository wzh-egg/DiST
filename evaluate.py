"""Evaluate the validation-selected DiST checkpoint on the AIR test set."""
from src.cli import parse_args


def main(args):
    import torch

    from src.engine import evaluate
    from src.experiment import setup
    from src.utils import write_json

    job, device, source, model = setup(args)
    checkpoint = torch.load(args.out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    epoch = checkpoint["epoch"]
    del checkpoint
    if job["compile"]:
        model = torch.compile(model, fullgraph=False, dynamic=False)
    result = evaluate(model, source, source.station.origins["test"], args.eval_batch, device)
    result["checkpoint_epoch"] = epoch
    write_json(args.out / "test_metrics.json", result)
    print(result["model"], flush=True)


if __name__ == "__main__":
    main(parse_args(__doc__))
