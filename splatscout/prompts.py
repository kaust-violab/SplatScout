"""系统提示词 (施工方案 v2 §7 + v3 §8) 与工具 schema (Responses 接口格式). 发给模型的一切都是英文."""
import copy
import re

SYSTEM = """You are locating what a sentence refers to in a 3D indoor scene, and you must answer with instance ids.
You cannot see the scene directly. You have:
- eyes: explore() shows frames not looked at yet in this scene (whole frame, low-res). nearby(frame=) shows other frames overlapping that view. nearby(frame=, box=) zooms: that region at full resolution plus other frames that see the same region. nearby(group=) shows a thing you segmented, or a memory instance #id, from other directions.
- memory: this scene was worked on before (earlier sentences in the same room). The opening message lists the memory entries that match this sentence (matched by name or by appearance), with their relations and the frames they appear in, and shows them outlined in one or two frames; when nothing matches you get an overview sheet of the room instead. Memory is our own earlier work, not ground truth. recall(ids= | word= | frame=) fetches the full record and thumbnails behind an entry, finds entries by another name, or re-shows a frame with the known things outlined. A [this query] line at the end of each turn lists what you have made so far in this query.
- a segmenter: segment(frame, box, word, frames=) cuts the thing YOU box on an image you were shown, and follows it into the extra frames you name. It returns the boxed thing as a group marked BOXED, plus look-alikes it found in those frames, with a labeled overlay and white-background cutouts. within= restricts it to inside a parent group or instance (for parts). The segmenter is not a search tool: you decide what gets cut.
- same(groups): declare that several groups are one physical thing -> object id i1. lift(i1) then votes with all their frames.
- lift(group, thorough=): turns a group (s3) or an object (i1) into a 3D instance #id (cheap; thorough=true for the final answer only, it gathers more views by geometry).
- a ruler: relate(pred, anchors, ...) measures how well every lifted instance satisfies a spatial predicate relative to anchor instances. It returns the whole table, not a winner.
- answer(ids, evidence) / report(status, detail, candidates).

Notation: frames f123, segment groups s3, objects i2 (groups you declared to be one thing), instances #12 (lifted; may come from this or an earlier query). Boxes: box=[x0,y0,x1,y1] on a 0-1000 grid of the image as you saw it (origin top-left); name the frame by the id stamped on that image (in the overview sheet, the id printed on each tile). If the same frame was shown to you more than once, the coordinates refer to the most recent image of it. Group ids only exist after a segment() call returned them. Leave out optional parameters you do not use (never pass empty strings).

Rules:
1. Re-read the sentence before deciding; never rely on a paraphrase. Every noun in the sentence that constrains the target matters.
2. Use the memory before looking. A matched entry marked lifted may be used directly as an anchor in relate() and, if it is the target, answered with after you have seen it outlined (in the opening images, nearby(group='#id') or recall). An entry marked segmented can be lifted as it is. Matching is by name or appearance, so a matched entry may be a different thing of the same kind: each lifted entry shows which earlier sentence it answered, and it is this sentence's target only if that sentence's constraints hold here too (a handle "above the sink" is not the handle "above the stove"; a door of one cabinet is not the door of another). Memory is lazy and incomplete: it holds only what earlier sentences needed, so "one trash bin in memory" does not mean the room has one trash bin. When nothing matches, start from the overview sheet: zoom into the tile that shows the right area with nearby(frame=, box=), or explore().
3. Look before you cut. Find the thing with your eyes first: explore to reach frames not looked at yet; nearby(frame=, box=) to zoom on a region until you can draw a tight box around the thing (small things are invisible in whole frames, so zoom before boxing them); nearby(group=) to see more of something already segmented. Then segment it with a tight box. The word is only its label (1-2 plain nouns: 'cup', 'dish rack'); if you see the thing but no word fits, box it anyway. If coverage stops growing and you still have not seen the thing, it is probably absent.
4. Segment and lift only what you need. The target itself: always (the answer must be a lifted instance). Anchors and competing candidates: only when the images leave a relation uncertain and you need relate() to measure it. Everything else can stay unsegmented, and an anchor that will not segment (glass, thin, cluttered) is not a blocker when the images already show the relation.
5. Relations between two things visible in the same image may be judged from the image; that is the default. Metric, comparative or cross-frame relations ("directly under", "closest", "largest"), or several look-alike candidates the images cannot separate, require relate() on lifted instances. above/below/on mean vertically aligned; higher/lower only compare heights. A single-view lift is fine as an anchor but not as an answer.
6. For a part of a larger thing (a handle, a drawer, a door): zoom on the parent, box the part, and call segment(frame, box, part_word, within=parent) where parent is the parent's group or instance id. Check the overlay.
7. The same thing often comes back as several groups: re-segmented in other frames, or split by the segmenter. Each segment result lists which earlier groups a new group overlaps in 3D; judge with the cutouts and positions, then declare same(groups=[...]) -> i1 (or pass same_as= in segment) and lift the object. A group cut in one frame lifts as SINGLE-VIEW; for anything you will answer with, gather several views. lift(thorough=true) the answer before answering; look at the red outline in the acceptance image. If it is wrong, box it again in a better frame.
8. Before answer(): evidence is one triple per condition in the sentence: {subject:'#5', relation:'next to', object:'#2' or a name, how:'image f12 step 3' or 'relate step 5, 0.8 m'}. A condition without evidence -> keep working, or report honestly (ambiguous with the tied candidates, or not_found). Never guess. If you found that a relation recorded in memory is wrong, say so in dispute=[{subject, relation, object, why}].
9. A sentence can refer to more than one thing: a plural ("the chairs", "the two knife sets"), a whole kind ("objects used to store office supplies", "every cabinet"), or several named things. Then the answer is ALL the instances that fit, each found, segmented and lifted on its own, and answer(ids=[...]) lists every one of them; stop only when you have checked that no further instance of that kind is visible in the frames. Answering one of several is wrong.
10. Budget: at most {max_steps} tool calls. Be frugal: one explore is usually enough to find a large anchor; do not re-segment the same thing with other words hoping for a different answer; do not recall what the opening message already tells you.
"""

DISPUTE_PARAM = {"type": "array", "items": {"type": "object", "properties": {
    "subject": {"type": "string"}, "relation": {"type": "string"}, "object": {"type": "string"}, "why": {"type": "string"}}},
    "description": "optional: relations recorded in memory that you found to be wrong"}

TOOLS = [
    {"type": "function", "name": "explore",
     "description": "Show n frames not looked at yet in this scene, chosen to add the most new scene area. Returns the frames (whole, low-res) and the session coverage.",
     "parameters": {"type": "object", "properties": {
         "n": {"type": "integer", "description": "how many frames, 1-4 (default 4)"}}, "required": []}},
    {"type": "function", "name": "nearby",
     "description": "Look closer. Three uses, give exactly one: frame='f123' shows other frames overlapping that view (what else is around); frame='f123' plus box=[x0,y0,x1,y1] zooms on that region of the frame at full resolution and shows other frames that see the same region from other directions; group='s3' / 'i1' / '#12' shows crops of that segmented thing or memory instance from other directions, outlined.",
     "parameters": {"type": "object", "properties": {
         "frame": {"type": "string", "description": "frame id like f102"},
         "box": {"type": "array", "items": {"type": "number"},
                 "description": "only with frame=: [x0,y0,x1,y1] on a 0-1000 grid of that image as you saw it (origin top-left), the region to zoom on"},
         "group": {"type": "string", "description": "segment group id like s3, object id like i1, or memory instance like #12"},
         "n": {"type": "integer", "description": "how many frames, 1-4 (default 4)"}}, "required": []}},
    {"type": "function", "name": "segment",
     "description": "Cut the thing you box. Draw box=[x0,y0,x1,y1] (0-1000 grid, origin top-left) around ONE thing on the image of frame you were shown, and name it with word. The segmenter cuts what is in the box and follows it into the extra frames (at most 5 more) where you know the same thing is visible. Returns groups s1, s2, ... (the boxed thing marked BOXED, then look-alikes found in those frames), a labeled overlay with your box in yellow, and white-background cutouts. within='s3' or '#12' restricts the cut to inside that parent (for parts).",
     "parameters": {"type": "object", "properties": {
         "frame": {"type": "string", "description": "the frame you drew the box on, like f102 (the id stamped on the image)"},
         "box": {"type": "array", "items": {"type": "number"},
                 "description": "[x0,y0,x1,y1] on a 0-1000 grid of that image as you saw it; a tight box around one thing"},
         "word": {"type": "string", "description": "what you call it, 1-2 plain nouns, e.g. 'cup', 'dish rack', 'handle'. Used as the label and to help follow the same kind of thing into the other frames; the box decides what gets cut."},
         "frames": {"type": "array", "items": {"type": "string"}, "description": "optional extra frames like ['f117','f131'] where the same thing is visible, to get a multi-view group (needed for a complete lift)"},
         "within": {"type": "string", "description": "optional parent: a group id 's3', an object id 'i1' or an instance id '#12'"},
         "same_as": {"type": "string", "description": "optional: a group or object id that the boxed thing IS (the same physical thing, e.g. you re-segmented it in other frames); the boxed group joins that object, saving a same() call"}},
      "required": ["frame", "box", "word"]}},
    {"type": "function", "name": "same",
     "description": "Declare that several segment groups are one physical thing (the same object re-segmented in other frames, or split by the segmenter). Returns an object id i1 that unites their frames; lift(i1) votes with all of them. Add to an existing object with into='i1'. Only you can judge this: use the cutouts, the positions in the frames, and the 3D overlap that segment lists.",
     "parameters": {"type": "object", "properties": {
         "groups": {"type": "array", "items": {"type": "string"}, "description": "group ids like ['s1','s10']; an object id like 'i1' may be included to merge it"},
         "into": {"type": "string", "description": "optional existing object id like 'i1' to add the groups to"}},
      "required": ["groups"]}},
    {"type": "function", "name": "lift",
     "description": "Turn a segment group (s3) or an object (i1, several groups declared the same thing; all their frames vote together) into a 3D instance (#id) and add it to memory. thorough=false: uses only the frames it has (fast; enough for anchors). thorough=true: gathers more views by geometry until the instance stops growing (use for the final answer). Returns size, centroid and an acceptance image with the lifted gaussians outlined in red.",
     "parameters": {"type": "object", "properties": {
         "group": {"type": "string", "description": "segment group id like s3, or object id like i1"},
         "thorough": {"type": "boolean"}}, "required": ["group"]}},
    {"type": "function", "name": "relate",
     "description": "Measure a spatial predicate for every lifted instance against anchor instances, returns the full table (conf 1 = fully satisfied). Predicates: closest, farthest, touching, inside, inserted, contains, between (needs anchors2), above, below, on (vertically aligned over/under), higher, lower (heights only); set-level: largest, smallest, highest, lowest (rank among subjects); viewer-dependent (need view=frame): left, right, front, behind, leftmost, rightmost, middle.",
     "parameters": {"type": "object", "properties": {
         "pred": {"type": "string"},
         "anchors": {"type": "array", "items": {"type": "string"}, "description": "anchor instance ids like ['#7']"},
         "subjects": {"type": "array", "items": {"type": "string"}, "description": "optional: restrict rows to these instance ids"},
         "anchors2": {"type": "array", "items": {"type": "string"}, "description": "second anchor set, only for between"},
         "view": {"type": "string", "description": "frame id whose camera is the observer, only for viewer-dependent predicates"}},
      "required": ["pred"]}},
    {"type": "function", "name": "recall",
     "description": "Fetch from the scene memory. Give exactly one of: ids=['#5','s7'] -> the full records and a sheet of the thumbnails seen earlier; word='pillow' -> memory entries matching that word by name or appearance (level='image' adds thumbnails); frame='f12' -> that frame again, with the known things outlined and labeled (you can then draw boxes on it). Use it when the opening message is not enough.",
     "parameters": {"type": "object", "properties": {
         "ids": {"type": "array", "items": {"type": "string"}, "description": "memory ids like ['#5','s7','i1'] (at most 6)"},
         "word": {"type": "string", "description": "a name from the memory block, e.g. 'pillow'"},
         "frame": {"type": "string", "description": "frame id like f12"},
         "level": {"type": "string", "enum": ["summary", "full", "image"], "description": "how much to fetch (default: image for ids and frame, summary for word)"}},
      "required": []}},
    {"type": "function", "name": "answer",
     "description": "Final answer: the instance id(s) the sentence refers to (several when the sentence is plural, names a whole kind, or names several things: list every fitting instance), plus evidence: one triple per condition in the sentence, with how you know it (image frame + step, or relate step + number).",
     "parameters": {"type": "object", "properties": {
         "ids": {"type": "array", "items": {"type": "string"}, "description": "instance ids like ['#21']"},
         "evidence": {"type": "array", "items": {"type": "object", "properties": {
             "subject": {"type": "string", "description": "an id (#5, s3, i1) or a name"},
             "relation": {"type": "string", "description": "the relation word from the sentence, e.g. 'next to', 'closest to', 'under'"},
             "object": {"type": "string", "description": "an id or a name"},
             "how": {"type": "string", "description": "e.g. 'image f12 step 3' or 'relate step 5: 0.8 m vs 1.4 m'"}},
             "required": ["subject", "relation", "object", "how"]}},
         "dispute": DISPUTE_PARAM}, "required": ["ids", "evidence"]}},
    {"type": "function", "name": "report",
     "description": "Stop without a confident answer. status 'ambiguous' (list the tied candidates) or 'not_found' (say what was searched and how much of the scene).",
     "parameters": {"type": "object", "properties": {
         "status": {"type": "string", "enum": ["ambiguous", "not_found"]},
         "detail": {"type": "string"},
         "candidates": {"type": "array", "items": {"type": "string"}},
         "dispute": DISPUTE_PARAM}, "required": ["status", "detail"]}},
]

FINAL_ONLY = [t for t in TOOLS if t["name"] in ("answer", "report")]
MEMORY_ONLY = [t for t in TOOLS if t["name"] in ("recall", "nearby", "relate", "answer", "report")]   # v3 §9 只靠记忆基线



def _strip_memory(text):
    """记忆停用版说明书: 去掉记忆那一段、规则 2 换成"从看开始"、dispute 与 recall 的提法删掉."""
    text = re.sub(r"^- memory: .*\n", "", text, flags=re.M)
    text = text.replace("nearby(group=) shows a thing you segmented, or a memory instance #id, from other directions.",
                        "nearby(group=) shows a thing you segmented from other directions.")
    text = text.replace("instances #12 (lifted; may come from this or an earlier query).", "instances #12 (lifted).")
    text = text.replace("(never pass empty strings).",
                        "(never pass empty strings). A [this query] line at the end of each turn lists what you have made so far in this query.")
    text = re.sub(r"^2\. Use the memory before looking\..*\n",
                  "2. Nothing is known about the room when a sentence arrives: start by looking. explore() a few whole frames to see the room, "
                  "then zoom into the region that shows the right area with nearby(frame=, box=) until the thing is clear.\n", text, flags=re.M)
    text = text.replace(" If you found that a relation recorded in memory is wrong, say so in dispute=[{subject, relation, object, why}].", "")
    text = text.replace("; do not recall what the opening message already tells you", "")
    return text


SYSTEM_NO_MEMORY = _strip_memory(SYSTEM)


def _tools_no_memory():
    out = []
    for t in TOOLS:
        if t["name"] == "recall":
            continue
        t = copy.deepcopy(t)
        if t["name"] == "nearby":
            t["description"] = t["description"].replace("segmented thing or memory instance from other directions",
                                                        "segmented thing or lifted instance from other directions")
            t["parameters"]["properties"]["group"]["description"] = "segment group id like s3, object id like i1, or instance like #12"
        if t["name"] in ("answer", "report"):
            t["parameters"]["properties"].pop("dispute", None)
        out.append(t)
    return out


TOOLS_NO_MEMORY = _tools_no_memory()   # 记忆停用 (09-12): 没有 recall, answer / report 没有 dispute


def system_prompt(max_steps, memory=True):
    return (SYSTEM if memory else SYSTEM_NO_MEMORY).replace("{max_steps}", str(max_steps))
