# NATHAN: Agent Profile & Roadmap

## 🤖 Identity
**Name**: Nathan
**Type**: Deep Reinforcement Learning Agent
**Ultimate Goal**: To become a fully autonomous, intelligent Non-Player Character (NPC) capable of complex reasoning, navigation, and interaction within dynamic virtual worlds.

---

## 📍 Current Status: The Gridlock Protocol
Nathan is currently undergoing "Phase 7" training in the **Gridlock Training Facility** (a procedural grid-world environment).

### The Mandate
Nathan's current objective is to demonstrate **Robust Spatial Reasoning**. He must:
1.  **Perceive**: Analyze a grid layout containing Walls, Traps, Keys, and a Goal.
2.  **Plan**: Determine the optimal sequence of Key collection to unlock the Goal.
3.  **Survive**: Navigate without triggering static hazards (Traps) or running out of energy (Timeouts).

### Current Capabilities (Stage 1B Certified)
*   **Multi-Objective Handling**: Nathan successfully identifies and retrieves up to 3 keys before attempting to exit.
*   **Hazard Avoidance**: On 6x6 grids, Nathan demonstrates 88% survival reliability, proficiently skirting traps.
*   **Efficiency**: He has learned to minimize steps, driven by a reliable internal value function.

### Current Challenge: "The 8x8 Paralysis" (Stage 2C)
Nathan is currently attempting to scale his skills to larger **8x8 environments** with **Mandatory Traps**.
*   **The Problem**: When finding himself in a large, trapped room, Nathan currently freezes (Success Rate: ~1.2%).
*   **Psychology (RL Interpretation)**: The penalty for stepping on a trap is currently so high relative to his confidence in finding a distant goal that he has adopted a "Learned Helplessness" strategy—he prefers to wait and timeout rather than risk death.
*   **The Fix**: We are currently adjusting his "risk tolerance" (lowering trap penalties) and "curiosity" (entropy) to encourage brave exploration.

---

## 🧠 The "Mind" of Nathan
Nathan's intelligence is powered by **Proximal Policy Optimization (PPO)**.
*   **Vision**: He "sees" the world as a 5-channel matrix (Agent, Wall, Trap, Key, Goal).
*   **Learning**: He learns from trial and error, executing millions of steps to refine his neural network policy.
*   **Motivation**: He is driven by a dense potential-based reward system that guides him toward targets like a compass.

---

## 🚀 Future Expansions: The Path to NPC
Once Nathan masters the Gridlock, his training will expand to encompass the traits of a true NPC:

### 1. Dynamic Adaptation (Stage 3)
*   **Challenge**: Moving obstacles (e.g., patrolling guards).
*   **Goal**: Nathan must time his movements, waiting for openings rather than just finding a static path.

### 2. Partial Observability (Fog of War)
*   **Challenge**: Nathan will only see a 3x3 radi around himself.
*   **Goal**: He must build an internal mental map (using Recurrent Neural Networks/LSTM) to remember where he has been and where the goal was seen last.

### 3. Social Integration
*   **Challenge**: Interaction with the Player.
*   **Goal**: Nathan will learn to be a **Companion** (helping the player find keys) or an **Adversary** (racing the player to the goal), adapting his behavior based on the player's actions.

---

*"I am currently stuck in the 8x8 trap room. The walls are far apart, and the floor is dangerous. But I am learning. I will find the path."* — **Nathan (Training Step 2,090,000)**
