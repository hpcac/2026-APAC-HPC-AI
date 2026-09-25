# 2026-APAC-HPC-AI

**Make the same math faster. Don't change it.**

The same line applies to every team.

---

## Addendum (2026-09-25)

This addendum is added below the original text. Nothing above it is removed. Where an earlier sentence in a task file no longer applies, that task file has its own "Addendum (2026-09-25)" section at the end that names the sentence it replaces.

Task-specific details:

- Qwen: [1. Qwen3 AI Task Rules.md → Addendum](1.%20Qwen3%20AI%20Task%20Rules.md#addendum-2026-09-25)
- OpenFOAM: [2. OpenFOAM HPC Task Rules.md → Addendum](2.%20OpenFOAM%20HPC%20Task%20Rules.md#addendum-2026-09-25)

### R1. What "the same math" means

- Qwen: the same model checkpoint (Qwen3-235B-A22B-Instruct-2507-FP8), the same precision (FP8 weights; KV in FP8 or higher), and no shortcut that changes what the model computes.
- OpenFOAM: the same case (65M mesh, `simpleFoam`, `fvSchemes`, `fvSolution.fixedIter`), double precision, and the same numerical work per step.

Making that computation run faster on the contest machines is the task. Changing the computation to get a faster number is not.

### R2. Three tiers

All work falls into one of three tiers. They apply to both tasks.

| Tier | What it is | Qwen (AI) | OpenFOAM (HPC) |
|---|---|---|---|
| **Basic** | Vanilla run. Required from every team. | Native FP8 end to end. Speculative decoding only if lossless. | Unmodified official v2512 source. Tune built-in controls only. |
| **Bonus** | Solid Basic tuning, **or** experimental work that the team proves correct itself. | Optional | Optional |
| **Prohibited** | Changing the math. Not allowed in any tier. | — | — |

**Basic.** Every team submits a Basic result for each task. The Basic result is what the organisers re-run. Each task's Addendum lists exactly what Basic allows.

**Bonus.** There are two ways to earn Bonus points:

1. **Recognition of Basic work.** Careful, well-explained Basic tuning can earn Bonus points on its own. You do not need to do experimental work to get Bonus points.
2. **Experimental work.** You may go beyond Basic (for example, change OpenFOAM source code, use other decomposition methods, or try new serving techniques). You must prove it is still the same math. Each task's Addendum says what counts as proof.

Bonus work never replaces the Basic result. Submit Basic first, then Bonus on top.

**Prohibited**, in every tier, including Bonus:

- Changing the math: a different model or case, a different numerical method, or less work per step or per token.
- Lowering floating-point precision: double to single or mixed precision in OpenFOAM; below FP8 in Qwen.
- Skipping AI computation, or skipping boundary or validity checks.

Everything already listed as prohibited in the task files stays prohibited.

### R3. How Bonus is judged

The burden of proof is on the team. Judges give Bonus points for experimental work when the team shows all four:

1. **What changed.** The exact change, as text (patch, script, flags, dictionaries).
2. **Why it is the same math.** A short argument a judge can check.
3. **Evidence.** The check named in the task's Addendum (OpenFOAM: Cd against your own pre-change baseline; Qwen: the accuracy check), with logs.
4. **Speedup against your own Basic result**, on the same machine.

A result the team cannot explain in the interview gets little credit, however fast it is. A clearly argued experiment that did not pay off can still be recognised.

Judges may also give extra Bonus points for Basic tuning they find especially solid.

### R4. Scoring

This addendum replaces "60%" and "40%" in the Presentation Guidelines and Scoring of both task files. Each task is now **70%** judges (Basic 20 + Bonus 15) and **30%** performance and re-verification (15), out of 50 points per task.

- The performance part ranks the Basic submission on its task's metric (Qwen: good output token/s; OpenFOAM: 4-node Average wall-clock time per time step). Platforms are not compared with each other.
- Qwen: a re-run that does not clear `good_request_fraction ≥ 0.90`, or scores below the reference deployment (`reference-scripts/07`, in the Application Notes), is scored at the reference deployment's score.
- One interview covers both tasks, as before.

### R5. Work already done

Anything the published task files allowed before this addendum is still allowed. Where this addendum moves a method from Basic to Bonus, the method stays legal; it is judged as Bonus work. You still need one Basic result per task.
