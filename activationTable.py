import torch.nn.functional as F
import torch
import parameters as p

class activationTable:
    def __init__(self, num_classes): # idx encodes class
        self.ref_activations = [0] * num_classes
        self.similar_counts = [1.0] * num_classes
        self.input_counts = [0] * num_classes
        self.class_layers = [0] * num_classes
        self.sim_threshold = torch.tensor(p.sim_threshold).to(p.device)
        self.is_ready = False
        
        
    def update_activation(self, label, activation, layer):
        label = int(label)
        count = self.input_counts[label]
        act = self.ref_activations[label]
        if len(act) == len(activation):  # update same layer running avg
            self.ref_activations[label] = ((act * count) + activation)/(count+1)
            self.input_counts[label] += 1
        else:  # layer has changed
            self.ref_activations[label] = activation
            self.input_counts[label] = 1
            self.class_layers[label] = label 
        
        
    def get_similarity(self, act1, act2):
        return F.cosine_similarity(act1.flatten(), act2.flatten(), 0)
    
    
    def find_most_similar(self, activation, layer):
        max_sim = 0
        max_idx = 0
        for i in range(len(self.class_layers)):
            #print("in func")
            if self.class_layers[i] == layer:
                sim = self.get_similarity(activation, self.ref_activations[i])
                #print(sim)
                if sim>max_sim:
                    max_sim = sim
                    max_idx = i
        
        #if max_sim == 0: # that layer doesn't have entry

        return max_idx, max_sim
    
    
    def check_ready(self):
        if self.is_ready == True:
            return True
        for el in self.ref_activations:
            if isinstance(el, int):
                return False
        self.is_ready == True
        return True

    
    def fill_table(self, activation, label):
        if self.check_ready() == False:
            if isinstance(self.ref_activations[label], int):
                self.ref_activations[label] = activation
                self.input_counts[label] = 1
                self.class_layers[label] = p.num_layers-1 # 0 indexed
                #print("Filled for "+str(label))
