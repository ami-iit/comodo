import random 
from enum import Enum, EnumMeta, auto

class NameChromosome(Enum): 
    LENGTH = auto()
    DENSITY = auto()
    MOTOR_INERTIA = auto()
    MOTOR_FRICTION = auto()
    JOINT_TYPE = auto()
    TSID_PARAMTERES = auto()
    MPC_PARAMETERS  = auto()
    PAYLOAD_LIFTING = auto()
    TIME_TRAJ_FEET = auto()
    TIME_TRAJ_PAYLOAD = auto()
    NOT_SET = auto()

class SubChromosome():
    def __init__(self):
        self.type = NameChromosome.NOT_SET
        self.isFloat = False 
        self.isDiscrete = False
        self.limits = None # 0 is low 1 is up 
        self.dimension = None 
        self.feasible_set = None 

class ChromosomeGenerator(): 

    def __init__(self) -> None:
        self.sub_chromosomes = [] 
        pass

    def add_parameters(self,subChr):
        self.sub_chromosomes.append(subChr)

    def create_float_param(self, limits, n_param):
        if(len(limits)>2):
            return[random.uniform(limits[i,0], limits[i,1]) for i in range(n_param)]
        return [random.uniform(limits[0], limits[1]) for _ in range(n_param)] 
    
    def create_discrete_param(self, feasible_set, n_param):
        return [random.choice(feasible_set) for _ in range(n_param)]

    def generate_chromosome(self):
        chromosome = []
        for item in self.sub_chromosomes:
            temp_chromosome= []
            if item.isFloat: 
                temp_chromosome = self.create_float_param(item.limits, item.dimension)
            if item.isDiscrete: 
                temp_chromosome = self.create_discrete_param(item.feasible_set, item.dimension)
            chromosome.extend(temp_chromosome)
        return chromosome

    def check_chromosome_in_set(self, ind): 
        idx_start_point = 0 
        for item in self.sub_chromosomes: 
            for idx in range(item.dimension): 
                if(item.isFloat): 
                    if(len(item.limits)>2):
                        ind[idx_start_point+idx] = min(max(ind[idx_start_point+idx], item.limits[idx,0]), item.limits[idx,1])
                    else:
                        ind[idx_start_point+idx] = min(max(ind[idx_start_point+idx], item.limits[0]), item.limits[1])
                
                elif(item.isDiscrete): 
                    if not(ind[idx_start_point+idx] in item.feasible_set):
                        ind[idx_start_point+idx] = random.choice(item.feasible_set)
            idx_start_point+=item.dimension
        return ind
        
    def get_chromosome_dict(self,ind): 
        idx_start_point = 0 
        return_dict = {}
        for item in self.sub_chromosomes: 
            return_dict.update({item.type:ind[idx_start_point:idx_start_point+item.dimension]})
            idx_start_point+=item.dimension
        return return_dict
    

    