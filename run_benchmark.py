from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)

@app.command()
def main(dataset: str = "data/hotpot_golden.json", out_dir: str = "outputs/golden_run", reflexion_attempts: int = 3) -> None:
    examples = load_dataset(dataset)
    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)
    import concurrent.futures
    from tqdm import tqdm
    
    react_records = []
    reflexion_records = []
    
    print("Running ReAct Agent...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        react_records = list(tqdm(executor.map(react.run, examples), total=len(examples)))
        
    print("Running Reflexion Agent...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        reflexion_records = list(tqdm(executor.map(reflexion.run, examples), total=len(examples)))
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode="mock")
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
