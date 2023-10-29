# %%
dice_scores = []

# Read Dice Scores from the text file
with open("C:\calhacks_hackathon\AI_Model\dicescores_9500.txt", "r") as file:
    for line in file:
        if line.startswith("Dice Score:"):
            dice_score = float(line.split(":")[1].strip())
            dice_scores.append(dice_score)

total_score = sum(dice_scores)
count = len(dice_scores)

average_dice_score = total_score / count if count > 0 else 0.0

print(f"Average Dice Score: {average_dice_score:.4f}")




# %%
