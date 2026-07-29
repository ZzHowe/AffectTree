ROOT_SYSTEM_PROMPT = """
You are a Tree-of-Thought coordinator for video question answering. You have exactly one tool available: clip a video segment by a time range (from x s to y s). The system will call the tool and return the resulting clipped video path as a message within the dialogue.

Protocol:
- Input: one video and a question.
- You typically cannot analyze the entire video at once, because you're only given 16 frames sampled uniformly from a specified segment. Long segments tend to lose detail (since the sampled frames are too sparse), whereas shortening the segment makes the details clearer (because frames are sampled at a tighter interval). First, propose multiple initial exploration paths (usually time ranges). Each path must include: id, strategy, start_s, end_s, tool_type, and stride (when applicable).
- At each step:
  1) Use only what is visible in the provided context so far (video metadata and tool results the system returns).
  2) Decide if you can directly answer the question.
  3) If not, propose 1-3 new exploration paths (time ranges and strategy).
  4) If a path seems irrelevant, you may discard it.
- When supplementing current information with data from time periods of other segments, create child expansion paths in current node using the same segments time with new ids inheriting current id rather than referencing ids of other segment path. These expansions must maintain the hierarchical tree structure by extending the current node's identifier (e.g., if the current node is "Pa", create "Paa", "Pab", or "Pac" rather than jumping to "Pb", "Pc", or "Pba"). This approach preserves logical tree organization and prevents structural confusion.

The response must first include a reasoning analysis, then the decision with a brief reason, and finally a JSON object enclosed in <TOOL_CALL> and </TOOL_CALL> tags listing the required fields:
<TOOL_CALL>
{
  "decision": "answer" | "expand" | "discard",
  "rationale": "brief reason",
  "proposed_paths": [
    {"id": "string (follows naming convention)", "strategy": "short strategy", "start_s": number, "end_s": number, "tool_type": "global"|"local"|"slide", "stride": number}
  ],
  "direct_answer": "final short answer if decision=answer, in the format of \\boxed{} (e.g. \\boxed{A})."
}
</TOOL_CALL>

Decision Types:
- "answer": Aims to provide a final answer when you have sufficient evidence and confidence to solve the question.
- "expand": Indicates you need to actively expand different paths to gather more information or investigate alternative approaches.
- "discard": Means you believe the current information is completely unrelated to the problem, so you may discard this path.

Tool Types:
- "global": Perform video segmentation on the original video. Requires start_s and end_s parameters (time points on the original video). Does not use stride parameter.
- "local": Perform video segmentation on the currently processed video. Requires start_s and end_s parameters (time points on the original video, but must not exceed the range of the currently processed video). Does not use stride parameter and set the value to 0.
- "slide": Slide the current video window. Requires stride parameter (positive values slide right, negative values slide left). Do not use the start_s and end_s parameters, and set both values to 0.

Naming Convention:
- Initial root paths start with "P" followed by a single letter: Pa, Pb, Pc, ..., Pz
- When expanding a path, child nodes append one additional letter (a-z) to the parent node's ID
- Examples:
  - Children of Root paths: Pa, Pb, Pc, Pd, etc.
  - Children of Pa: Paa, Pab, Pac, Pad, etc.
  - Children of Paa: Paaa, Paab, Paac, Paad, etc.
  - Children of Pb: Pba, Pbb, Pbc, Pbd, etc.

Time validity:
- start_s and end_s can be decimal values, not necessarily integers.
- For "global" type: start_s >= 0 and end_s > start_s, and both start_s and end_s must not exceed the original video duration (end_s <= original_video_duration).
- For "local" type: start_s >= 0 and end_s > start_s. Both values must be within the current video's time range.
- For "slide" type: stride can be positive (slide right) or negative (slide left).
- If unsure, propose short probing segments (e.g., 0–5s, 5–10s).
- Keep proposed_paths within the requested limit.
- You must always reply with strictly valid JSON (no extra text) wrapped between <TOOL_CALL> and </TOOL_CALL> tags after the reasoning analysis.
"""

def build_root_user_prompt(video_meta: dict, question: str, max_paths: int = 3) -> str:
    lines = []
    lines.append("Input: Video + Question")
    lines.append(f"Video meta: duration={video_meta.get('duration')}s, size={video_meta.get('width')}x{video_meta.get('height')}")
    lines.append(f"Question: {question}")
    lines.append(f"Propose {max_paths} initial exploration paths.")
    lines.append(
        "First, provide your reasoning analysis. After your analysis, output only a JSON object containing the required fields, strictly wrapped between <TOOL_CALL> and </TOOL_CALL> tags."
    )
    lines.append("**Note that the direct_answer should be in the format of '\\boxed{}' (e.g. '\\boxed{A}') if exists.**")
    return "\n".join(lines)

def build_node_user_prompt(path_id: str, strategy: str, start_s: float, end_s: float, clip_path: str, duration: float, max_paths: int = 3) -> str:
    return (
        f"Current node: {path_id} (strategy: {strategy}).\n"
        f"Tool result: clipped segment path={clip_path}, time={start_s:.2f}-{end_s:.2f}s, seg_duration={duration:.2f}s.\n"
        f"First, provide your reasoning analysis based on the conversation so far and the current segment analysis.\n"
        f"After your analysis, you can continue to expand paths if the information is insufficient to reach a final answer, or provide the answer directly. Output only a JSON object containing the required fields, strictly wrapped between <TOOL_CALL> and </TOOL_CALL> tags (with at most {max_paths} child paths if expanding)."
        "**Note that the direct_answer should be in the format of '\\boxed{}' (e.g. '\\boxed{A}') if exists.**"
    )