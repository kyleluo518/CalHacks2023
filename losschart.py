# %%
import matplotlib.pyplot as plt

# Define the path to the loss file
loss_file_path = r'C:\calhacks_hackathon\ImageSegmentation\loss.txt'

# Read loss values from the text file, skipping the first line (header)
with open(loss_file_path, "r") as file:
    next(file)  # Skip the first line (header)
    loss_values = [float(line.strip()) for line in file]

# Create a list of iteration numbers (0, 1, 2, ...)
iterations = list(range(len(loss_values)))

# Create a line plot
plt.plot(iterations, loss_values, marker='o', linestyle='-')

# Set labels and title
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Over Iterations')

# Show the plot
plt.show()




# %%