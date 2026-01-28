import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500))) 

def sigmoid_derivative(a):
    return a * (1 - a)

def tanh(z):
    z = np.clip(z, -500, 500)
    
    e_pos = np.exp(z)
    e_neg = np.exp(-z)
    
    return (e_pos - e_neg) / (e_pos + e_neg)


def tanh_derivative(a):
    return 1 - a**2

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-8)  

def cross_entropy_loss(y_true, y_pred):
    eps = 1e-8
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))


class NeuralNetwork:
    def __init__(self, X, y, hidden_layers=(8,6), output_size=3, activation='tanh', use_bias=True, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.X = X
        self.y = y
        self.layer_sizes = [X.shape[1]] + list(hidden_layers) + [output_size]
        self.use_bias = use_bias
        self.train_history = {'loss': [], 'accuracy': []}
        self.test_history = {'loss': [], 'accuracy': []}

     
        self.thetas = []
        for i in range(len(self.layer_sizes)-1):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i+1]
        
            limit = np.sqrt(6 / (in_size + out_size))
            if self.use_bias:
                theta = np.random.uniform(-limit, limit, (in_size + 1, out_size))
            else:
                theta = np.random.uniform(-limit, limit, (in_size, out_size))
            self.thetas.append(theta)

    
        if activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            raise ValueError("activation must be 'tanh' or 'sigmoid'")

    def feedforward(self, X_input=None, return_all=False):
        if X_input is None:
            A = self.X
        else:
            A = X_input
        activations = [A]
        Z_s = []

        for i, theta in enumerate(self.thetas):
            if self.use_bias:
                A_with_bias = np.hstack([np.ones((A.shape[0], 1)), A])
            else:
                A_with_bias = A
            Z = np.dot(A_with_bias, theta)
            Z_s.append(Z)

            if i == len(self.thetas) - 1:
                A = softmax(Z)
            else:
                A = self.activation(Z)

            activations.append(A)

        if return_all:
            return activations, Z_s
        return A

    def backprop(self, l_rate):
        activations, _ = self.feedforward(return_all=True)
        delta = activations[-1] - self.y

        dThetas = [None] * len(self.thetas)

        for l in reversed(range(len(self.thetas))):
            A_prev = activations[l]
            if self.use_bias:
                A_prev = np.hstack([np.ones((A_prev.shape[0], 1)), A_prev])

      
            dThetas[l] = np.dot(A_prev.T, delta) / A_prev.shape[0]

            if l != 0:
                theta_no_bias = self.thetas[l][1:] if self.use_bias else self.thetas[l]
                delta = np.dot(delta, theta_no_bias.T) * self.activation_derivative(activations[l])

        for i in range(len(self.thetas)):
            self.thetas[i] -= l_rate * dThetas[i]

    def train(self, X_test=None, y_test=None, y_test_raw=None, epochs=200, l_rate=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        for epoch in range(1, epochs+1):
            self.backprop(l_rate)
            

            train_output = self.feedforward()
            train_loss = cross_entropy_loss(self.y, train_output)
            train_pred = np.argmax(train_output, axis=1)
            train_true = np.argmax(self.y, axis=1)
            train_acc = np.mean(train_pred == train_true)
            
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            
    
            if X_test is not None and y_test is not None:
                test_output = self.feedforward(X_test)
                test_loss = cross_entropy_loss(y_test, test_output)
                test_pred = np.argmax(test_output, axis=1)
                
                if y_test_raw is not None:
                    test_acc = np.mean(test_pred == y_test_raw)
                else:
                    test_true = np.argmax(y_test, axis=1)
                    test_acc = np.mean(test_pred == test_true)
                
                self.test_history['loss'].append(test_loss)
                self.test_history['accuracy'].append(test_acc)
           
                if epoch % 50 == 0 or epoch == 1:
                    print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    def predict(self, X_input):
        output = self.feedforward(X_input)
        return np.argmax(output, axis=1), output

def confusion_matrix_multiclass(y_true, y_pred, num_classes=3):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

def prepare_data(df, random_seed=42):
    np.random.seed(random_seed)
    species_list = ['Adelie', 'Chinstrap', 'Gentoo']
    features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass', 'OriginLocation']

    data = df[df['Species'].isin(species_list)].copy()
    
    numeric_cols = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
    
    for sp in species_list:
        mask = data['Species'] == sp
        for col in numeric_cols:
            species_mean = data.loc[mask, col].mean()
            data.loc[mask & data[col].isna(), col] = species_mean
    
    mode_loc = data['OriginLocation'].mode()[0]
    data['OriginLocation'] = data['OriginLocation'].fillna(mode_loc)
    
    data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    label_map = {sp: i for i, sp in enumerate(species_list)}
    data['label'] = data['Species'].map(label_map)

    train_list, test_list = [], []
    for sp in species_list:
        sub = data[data['Species'] == sp].copy()
        sub = sub.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
    
        n_samples = len(sub)
        if n_samples < 50:
            print(f"Warning: {sp} has only {n_samples} samples. Using 60% for training, 40% for testing.")
            train_size = int(n_samples * 0.6)
            train_list.append(sub.iloc[:train_size])
            test_list.append(sub.iloc[train_size:])
        else:
            train_list.append(sub.iloc[:30])
            test_list.append(sub.iloc[30:50])

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    X_train = train_df[features].values.astype(float)
    X_test = test_df[features].values.astype(float)
    y_train_raw = train_df['label'].values.astype(int)
    y_test_raw = test_df['label'].values.astype(int)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    y_train = np.zeros((len(y_train_raw), 3))
    y_train[np.arange(len(y_train_raw)), y_train_raw] = 1
    y_test = np.zeros((len(y_test_raw), 3))
    y_test[np.arange(len(y_test_raw)), y_test_raw] = 1

    return X_train, y_train, X_test, y_test, y_train_raw, y_test_raw, mean, std


class PenguinClassifierGUI:
    def __init__(self, root, df):
        self.root = root
        self.root.title("Penguin Species Classifier - Neural Network")
        self.root.geometry("900x550")
        
        self.df = df
        self.nn = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_test_raw = None
        self.mean = None
        self.std = None
        self.species_list = ['Adelie', 'Chinstrap', 'Gentoo']
        
        self.create_widgets()
    
    def create_widgets(self):
     
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
  
        config_frame = ttk.LabelFrame(main_frame, text="Neural Network Configuration", padding="10")
        config_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
 
        ttk.Label(config_frame, text="Number of Hidden Layers:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.num_layers_var = tk.IntVar(value=1)
        ttk.Spinbox(config_frame, from_=1, to=100, textvariable=self.num_layers_var, width=10, 
                    command=self.update_neurons_fields).grid(row=0, column=1, sticky=tk.W, padx=5)
        
    
        self.neurons_frame = ttk.Frame(config_frame)
        self.neurons_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.neurons_entries = []
        self.update_neurons_fields()
        

        ttk.Label(config_frame, text="Learning Rate (η):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.lr_var = tk.DoubleVar(value=0.05)
        ttk.Entry(config_frame, textvariable=self.lr_var, width=15).grid(row=2, column=1, sticky=tk.W, padx=5)
        
    
        ttk.Label(config_frame, text="Number of Epochs (m):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.epochs_var = tk.IntVar(value=200)
        ttk.Entry(config_frame, textvariable=self.epochs_var, width=15).grid(row=3, column=1, sticky=tk.W, padx=5)
        
       
        self.bias_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(config_frame, text="Add Bias", variable=self.bias_var).grid(row=4, column=0, sticky=tk.W, pady=5)
        
   
        ttk.Label(config_frame, text="Activation Function:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.activation_var = tk.StringVar(value="sigmoid")
        activation_frame = ttk.Frame(config_frame)
        activation_frame.grid(row=5, column=1, sticky=tk.W, padx=5)
        ttk.Radiobutton(activation_frame, text="Sigmoid", variable=self.activation_var, 
                       value="sigmoid").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(activation_frame, text="Hyperbolic Tangent", variable=self.activation_var, 
                       value="tanh").pack(side=tk.LEFT, padx=5)

        ttk.Button(config_frame, text="Train Neural Network", command=self.train_network).grid(
            row=6, column=0, columnspan=2, pady=10)
        

        class_frame = ttk.LabelFrame(main_frame, text="Single Sample Classification", padding="10")
        class_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
      
        default_values = [40.0, 18.0, 190.0, 3500.0, 0.0]
        features = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 
                   'Body Mass (g)', 'Origin Location (0=Torgersen, 1=Biscoe, 2=Dream)']
        self.feature_vars = []
        
        for i, (feature, default) in enumerate(zip(features, default_values)):
            ttk.Label(class_frame, text=f"{feature}:").grid(row=i, column=0, sticky=tk.W, pady=3)
            var = tk.DoubleVar(value=default)
            ttk.Entry(class_frame, textvariable=var, width=20).grid(row=i, column=1, sticky=tk.W, padx=5)
            self.feature_vars.append(var)
        
        
        ttk.Button(class_frame, text="Classify Sample", command=self.classify_sample).grid(
            row=len(features), column=0, columnspan=2, pady=10)
        
      
        self.result_label = ttk.Label(class_frame, text="", font=('Arial', 12, 'bold'), foreground='green')
        self.result_label.grid(row=len(features)+1, column=0, columnspan=2, pady=5)
        
    
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
    
    def update_neurons_fields(self):
   
        for widget in self.neurons_frame.winfo_children():
            widget.destroy()
        self.neurons_entries.clear()
        
        num_layers = self.num_layers_var.get()
        ttk.Label(self.neurons_frame, text="Neurons per layer:").grid(row=0, column=0, sticky=tk.W)
        
        for i in range(num_layers):
            ttk.Label(self.neurons_frame, text=f"Layer {i+1}:").grid(row=i+1, column=0, sticky=tk.W, padx=(20, 5))
            var = tk.IntVar(value=10)
            entry = ttk.Entry(self.neurons_frame, textvariable=var, width=10)
            entry.grid(row=i+1, column=1, sticky=tk.W, padx=5)
            self.neurons_entries.append(var)
    
    def train_network(self):
        try:
   
            self.X_train, self.y_train, self.X_test, self.y_test, _, self.y_test_raw, self.mean, self.std = prepare_data(self.df)
            
  
            hidden_layers = tuple([var.get() for var in self.neurons_entries])
            learning_rate = self.lr_var.get()
            epochs = self.epochs_var.get()
            use_bias = self.bias_var.get()
            activation = self.activation_var.get()
            
    
            self.nn = NeuralNetwork(
                self.X_train,
                self.y_train,
                hidden_layers=hidden_layers,
                output_size=3,
                activation=activation,
                use_bias=use_bias,
                seed=42
            )
            
            print("\nStarting training...")
            self.nn.train(
                X_test=self.X_test,
                y_test=self.y_test,
                y_test_raw=self.y_test_raw,
                epochs=epochs,
                l_rate=learning_rate,
                seed=42
            )
            
            y_pred_test = self.nn.predict(self.X_test)[0]
            cm = confusion_matrix_multiclass(self.y_test_raw, y_pred_test, num_classes=3)
            accuracy = np.mean(y_pred_test == self.y_test_raw)
            
            train_pred = self.nn.predict(self.X_train)[0]
            train_true = np.argmax(self.y_train, axis=1)
            train_accuracy = np.mean(train_pred == train_true)
            
            result_msg = f"Training Complete!\n\n"
            result_msg += f"Final Train Accuracy: {train_accuracy*100:.2f}%\n"
            result_msg += f"Final Test Accuracy: {accuracy*100:.2f}%\n\n"
            result_msg += "Confusion Matrix:\n"
            result_msg += "            Predicted\n"
            result_msg += "          A    C    G\n"
            for i, row in enumerate(cm):
                species_label = self.species_list[i][0]
                result_msg += f"Actual {species_label}  {row[0]:3d}  {row[1]:3d}  {row[2]:3d}\n"
            
            messagebox.showinfo("Training Complete", result_msg)
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def classify_sample(self):
        try:
            if self.nn is None:
                messagebox.showwarning("Warning", "Please train the network first!")
                return
            
      
            sample = np.array([[var.get() for var in self.feature_vars]])

            sample_normalized = (sample - self.mean) / self.std
            

            prediction, probabilities = self.nn.predict(sample_normalized)
            predicted_species = self.species_list[prediction[0]]

            result_text = f"Predicted Species: {predicted_species}\n\nProbabilities:\n"
            for i, species in enumerate(self.species_list):
                result_text += f"{species}: {probabilities[0][i]*100:.2f}%\n"
            
            self.result_label.config(text=f"✓ Predicted: {predicted_species}")
            
            messagebox.showinfo("Classification Result", result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    df = pd.read_csv('penguins.csv')

    location = {'Torgersen': 0, 'Biscoe': 1, 'Dream': 2}
    df['OriginLocation'] = df['OriginLocation'].map(location)
    
    

    root = tk.Tk()
    app = PenguinClassifierGUI(root, df)
    root.mainloop()