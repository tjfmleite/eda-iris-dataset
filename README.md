# Exploratory Data Analysis (EDA) on Iris Dataset

This project performs Exploratory Data Analysis (EDA) on the **Iris Dataset**, one of the most popular datasets in machine learning and statistics. The goal is to understand the data structure, visualize relationships between variables, and generate insights about the dataset.

## **Dataset**
The Iris dataset contains 150 samples of iris flowers from three species: Setosa, Versicolor, and Virginica. Each sample has the following features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

The target variable is the species of the iris flower.

The dataset is downloaded directly from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).

---

## **Project Features**
1. **Data Download**: The script automatically downloads the Iris dataset.
2. **Data Cleaning**: Basic checks for missing values and data consistency.
3. **Descriptive Statistics**: Summary statistics for all numeric columns.
4. **Visualizations**:
   - Pair plot for exploring relationships between features.
   - Heatmap for correlation analysis.
   - Boxplots for feature distributions by species.
   - Histograms for feature distributions.
5. **Grouped Statistics**: Mean and standard deviation for each feature grouped by species.
6. **Export Results**: Saves grouped statistics to a CSV file for further analysis.

---

## **Setup and Installation**
### **Requirements**
- Python 3.8 or later
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`

### **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/tjfmleite/eda-iris-dataset.git
   cd eda-iris-dataset
