# %%
# Open the data.txt file for reading
with open("C:\calhacks_hackathon\DataExtraction\data.txt", "r") as file:
    # Initialize an empty list to store the loss values
    loss_values = []

    # Iterate through each line in the file
    for line in file:
        # Check if the line contains "Loss=" to identify lines with loss information
        if "Loss=" in line:
            # Extract the loss value (assuming it's after "Loss=")
            loss_str = line.split("Loss=")[1].strip()
            # Convert the loss value to a float and append it to the list
            loss_values.append(float(loss_str))

# Print the extracted loss values
for idx, loss in enumerate(loss_values):
    print(f" {loss:.8f}")


# %%