#!/usr/bin/env python3
"""Quick test to validate PPO reward functions are working."""

import sys
sys.path.insert(0, '/home/rl-group10/training_scripts/open-rs/src')

from open_r1.rewards import format_reward, tag_count_reward

# Test 1: Good completion with tags
good_completion = """<think>
Let me solve this step by step. We have a square with vertices at (0,0), (0,3), (3,3), and (3,0).
The area of the square is 3 * 3 = 9.
We need to find the probability that x + y < 5.
The region where x + y < 5 is below the line y = 5 - x.
</think>

<answer>
The line x + y = 5 passes through (2, 3) and (3, 2).
The area where x + y < 5 within the square is a triangle and a trapezoid.
Actually, the area is half of the region below the line.
The probability is 7/18.
</answer>"""

# Test 2: Bad completion without proper tags
bad_completion = """The probability is 7/18 because the square has area 9 and the region where x+y<5 has area 3.5."""

# Test 3: Partial tags
partial_completion = """<think>
Let me think about this problem.
</think>

The answer is 7/18."""

print("=" * 80)
print("Testing format_reward function")
print("=" * 80)

# Test format reward
formatted_good = [{"role": "assistant", "content": good_completion}]
formatted_bad = [{"role": "assistant", "content": bad_completion}]
formatted_partial = [{"role": "assistant", "content": partial_completion}]

reward_good_format = format_reward([formatted_good])
reward_bad_format = format_reward([formatted_bad])
reward_partial_format = format_reward([formatted_partial])

print(f"Good completion format reward: {reward_good_format}")
print(f"Bad completion format reward: {reward_bad_format}")
print(f"Partial completion format reward: {reward_partial_format}")

print("\n" + "=" * 80)
print("Testing tag_count_reward function")
print("=" * 80)

reward_good_tags = tag_count_reward([formatted_good])
reward_bad_tags = tag_count_reward([formatted_bad])
reward_partial_tags = tag_count_reward([formatted_partial])

print(f"Good completion tag count reward: {reward_good_tags}")
print(f"Bad completion tag count reward: {reward_bad_tags}")
print(f"Partial completion tag count reward: {reward_partial_tags}")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"Good completion should have high rewards: format={reward_good_format[0]:.2f}, tags={reward_good_tags[0]:.2f}")
print(f"Bad completion should have low rewards: format={reward_bad_format[0]:.2f}, tags={reward_bad_tags[0]:.2f}")
print(f"Partial completion should have medium rewards: format={reward_partial_format[0]:.2f}, tags={reward_partial_tags[0]:.2f}")

if reward_good_format[0] > reward_bad_format[0] and reward_good_tags[0] > reward_bad_tags[0]:
    print("\n✓ Rewards are working correctly!")
else:
    print("\n✗ Rewards are NOT working as expected!")
