# Field Notebook — a children's game built on the Canon

> Concept draft · 2026-11-06 · for fun, not yet a spec

## One sentence

A game where kids aged 6–10 discover the rules of nature by predicting what will happen, running the experiment, and writing the rule in their own words, and where every rule they find is a real node in the Canon.

## The design rule that makes it work

**The game never states a law.** It sets up a world, asks the child to guess, runs the simulation, then lays a real dataset over the top ("here's what real birds on real islands did"). The child writes the rule on a notebook page in their own words. The notebook is theirs: it exports as a PDF they can print, and nothing they write leaves the device. That is the ownership thesis at kid scale.

The second rule: **every level ends outside.** Each law comes with a ten-minute activity in a park or a backyard that reproduces it with a pencil and four bits of string.

## Five levels, five real patterns

| Level | What the child does | The law underneath | Outside activity |
|---|---|---|---|
| **Islands** | Draws islands of different sizes in a sea; creatures raft over on logs; some islands are near the mainland, some far. Before counting, the child guesses how many kinds each island will keep. | Species–area relationship and island biogeography (nodes 5, 10). Big islands keep more; near islands get more; it is a curve, not a line. | Four string squares on the lawn: 1 step, 2 steps, 4 steps, 8 steps across. Count the kinds of plants in each. Does doubling the square double the kinds? |
| **Flock or scatter** | Chooses whether a creature travels alone, in pairs, or in big flocks. Then drops 20 sample squares on the map and counts. The "wobble meter" shows how different the counts are. | Taylor's power law (node 1). Clumpy animals make wild counts. The slope on log paper is the rule, but the child only sees "wobble grows faster when they flock". | Count pigeons versus sparrows in four spots at the park. Which count is more all-over-the-place? |
| **Feed the zoo** | A mouse, a cat, a dog, a horse, an elephant, each hungry. The child guesses how many berries each needs per day. The game weighs them and shows the real answer. | Kleiber's law (node 2). The elephant needs far fewer berries per kilo than the mouse. The child notices the "per kilo" trick themselves. | Hold a hamster and a dog (or find videos). Whose heart is racing? Count beats for 15 seconds. |
| **Stack the food chain** | Builds a pyramid: grass blocks, rabbit blocks, fox blocks. Too many foxes per rabbit and it topples. | Lindeman's 10% rule (node 19), labelled "rule of thumb" in the Canon, so the game says "about a tenth" and shows that real numbers vary. | Find one fox-sized predator's territory on a map and count how many rabbit-sized things it would need. |
| **The chase** | A see-saw: hares on one end, lynx on the other. The child tips it and watches the numbers chase each other round. | Predator–prey cycles (node 9). The overlay is the Hudson's Bay fur records, nearly a century of real data. | Nothing to count. The outside activity is a story: ask a grandparent what animals used to be common here and aren't now. |

## Why it is not a quiz

Each level has one guess, one experiment, one overlay, one notebook page. No points, no timers, no streaks. The only collectible is the notebook. A level is "done" when the child has written a sentence that starts with "I found out that". The game stores nothing but that sentence and the child's drawing.

## What the Canon gives the game

- The levels are the ✓ and ◐ nodes from docs/CANON.md, so when a check runs on real NEON data, the game's overlay updates. The game is a consumer of the library, the way the field-guide site is.
- Status words translate for kids: `supported` becomes "scientists are pretty sure", `contested` becomes "scientists are still arguing", `open_problem` becomes "nobody knows yet, maybe you". The last one is the best page in the notebook. Level six, if there is one, is May's paradox: "Big messy worlds should fall over. They don't. Why?" No experiment. Just the question.

## Hardest design problem

Keeping the overlay honest. Real data is messier than the simulation, and the temptation is to smooth it so the curve looks clean. Don't. The notebook page has two boxes: "what I guessed" and "what really happened". The gap between them is the whole lesson, and it is the same gap the Canon exists to record.

## Prototype path

1. Build "Islands" alone as one HTML page using the field-guide site's design tokens. Canvas, drag to draw islands, one Count button, one overlay from GBIF island species counts (node 5's check output, once it exists).
2. Playtest with three kids. Watch whether they write the sentence.
3. If they do, build "Flock or scatter" using the same Taylor's law simulator already in the field-guide hero.

A later version could ship as a tool in the MCP server too, so a classroom agent can open a level by name. That is a long way off.

## A note

The prime-numbers story from the podcast is the model here: a toddler with 19 blocks who can't make a rectangle has discovered primes without the word. Nobody told him. The block rolled under the couch. The best version of this game is one where the block rolls under the couch on purpose.
