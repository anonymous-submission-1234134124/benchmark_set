from dataclasses import dataclass, field


@dataclass
class ItemToRun:
    idx: int
    dp_id: str
    file_path: str
    replace_content: str
    method_name: str
    start_line: int
    end_line: int
    tests: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert the ItemToRun instance to a dictionary for JSON serialization."""
        return {
            "idx": self.idx,
            "dp_id": self.dp_id,
            "file_path": self.file_path,
            "replace_content": self.replace_content,
            "method_name": self.method_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "tests": self.tests,
        }
