Seems like prohibitively expensive; potentially expands context window but does not decrease the cost
outlier runs can take a very long time. 


Idea: Hm what if we use much cheaper openai models. Like thesis being cansubsittue older cheaper models for newer ones. So not as cost prohibitve.

- test with openai nano-> 5 million tokens = about 1 dollar

Intuitivley not sure what adding more layers will actually do, or how it will actually help. Since main benefit of orchestrator + sub calls means subcalls get stupid junior engineer tasks. But no need to break it down further. Just one brain is enough, can just add more iterations for longer complex tasks and more children subcalls if context increases
- parent cost linear (history bloat)
- Children cost grows with context


3 main knobs (many more)
- subcalls per iteration
- number of iterations
- Depth

Hm maybe since right now s-niah is fine 100% accuracy for both gpt 5 and rlm gpt this might be low hanging fruit to try to hit billion? Might only need minor changes


TODO
- run basic oolong (probalby pairs cause cheaper). Test with default paper.
- Try to build out a test harness i can just let go brr




OOLONG PAIRS
- Quadratic difficulty basically means impossible after certain context length for regular models. 
- think about a real world example of this



Benchmarks
1. retrieval (constant time). NIAH, RULER. only testing retrieval not reasoning
2. aggregation (linear). 
3. Relationsal (qudratic)


SNIAH- solved to infinity
- Can basically cheat on 1 billoin context window with repl search. 


OOLONG
- wait what if i have the parent be a smart one but have the children be really cheap dumb, and they solve one at a tiem? 
-PROBLEM: higher number of counts get harder bc score =0.75 ^ difference between actual count vs ai count; note difference when it si like 10 vs 15 vs 1000000 vs 1000100 (latter is basically score of 0).  At small context lengths (few hundred items), being off by 5 is plausible and you get partial credit. At large context lengths (thousands of items), even 95% classification accuracy means you're off by dozens, and the score collapses to zero regardless of how good your approach is.




Main problem with oolong and oolong pairs is children missclassifying- one line misclassify snowballs. Hm if children could talk with one antoher maybe they could do better?
- or maybe a two pass method -> store errors elsewhere and recheck once theyhit capacity?
- Or knowledge grpah?





# BIG PROBLEM:

  - S-NIAH: Solved trivially with code. Useless.
  - OOLONG counting: At large scale, scoring collapses. Hard to show differential improvement.
  - OOLONG-Pairs: Better — F1 scoring is more forgiving, and pairs genuinely benefit from shared state. But still limited by classification accuracy.
  - Scaling context to 1B: Prohibitive cost/time, and scoring breaks down anyway.

  Wait also oolong is a pretty bad benchmark; only good for decomposable aggregation; classify each item independently coutn report. 
  - (eg. won't be able to do smth like explain the general mood of this novel.)
  -  "Explain the general mood of this novel" requires:
  - Understanding how mood shifts across chapters
  - Tracking character arcs that develop over the whole text
  - Recognizing themes that are built gradually, not stated in any single paragraph
  - Synthesizing tone, word choice, narrative structure holistically
  - can't split novel into 50 chunks then ask mood and average them.

Approaches
- improve the history with existing benchmark -> get a better score on these benchmarks
    Split int o two, like a plan variable so not affected by compaction
    hard to do probably since RLM is kind of designed for this sort of count-then aggregate model. Kind of cherry picked oolong.
- new benchmark? which requires cross call memory? 
    - the novel example
    - infinite bench and misque
    - oolong is pretty cherry picked (niche) for the strengths of the RLM.
 - 1 billion (not possible) (see screenshot)


Next steps;
1. replicate 
2. Wonder if i just do a double pass for each child if it makes it better. See if there's existing papers on this (like gpt running twice, more likely to get it right?). Wang et all self consistency. RUn prompt n times take majority vote. Would this be described in the prompt only?  for sub calls do like multiple and take majority vote?
    same amount of time roughly (assuming async firing)
    might drastically improve oolong score (bc absolute difference). So one less error is big deal. 
3. play around with prompts to see if i can get a better score (single shot vs average? figure this out)
4. I suspect comparing RLM to regular GPT5 not much difference if anythign worse on the other novel benches.
    **CONFIRMED on MuSiQue (3/27/2026):** Ran 5 MuSiQue 2-hop tasks. Vanilla GPT-5 scored 0.80 EM / 0.933 F1. RLM (GPT-5 + GPT-5-mini) scored 0.60 EM / 0.600 F1. RLM was also 14x slower (196s vs 14s). OOLONG is cherry-picked — on multi-hop QA that requires chaining facts across paragraphs, vanilla GPT-5 beats RLM.


Graph as plan-> but use ablations with just regular key value and also rag. 



wait also oolong scoring is not a good idea feels more artificial. 


Rigure out if it actually makes sense to improve depth. -> ANOTHER ADDITIONAL STEP

Figure out rlm page in the paper. 
figure out what they mean by posttraining

extrmely stochastic might only put one line for child call vs like whole context