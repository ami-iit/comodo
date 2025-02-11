import pandas as pd
import os
import datetime
import fcntl

class DatabaseFitnessFunction:
    def __init__(self, name_database: str) -> None:
        self.name_database_with_ext = name_database + ".csv"
        try:
            self.database = pd.read_csv(self.name_database_with_ext)
        except:
            self.create_empty_csv_database()

    def create_empty_csv_database(self) -> None:
        self.database = self.get_empty_dataframe()
        self.database.to_csv(self.name_database_with_ext, index=False)

    def get_fitness_value(self, chromosome: list) -> float:
        a = self.database["chromosome"]
        str_chromosome = ','.join([str(x) for x in chromosome])
        if (
            len(self.database["chromosome"]) > 0
            and len(self.database[self.database["chromosome"] == str_chromosome]) > 0
        ):
            fitness_value = self.database[
                self.database["chromosome"] == str_chromosome
            ]["fitnessValue"].min()
        else:
            fitness_value = None
        return fitness_value


    def get_generation_fitness(self, generation: int) -> list:
        a = self.database["generation"]
        if len(self.database["generation"]) > 0:
            fitness_pop = self.database[self.database["generation"] == generation][
                "fitnessValue"
            ]
            pop = self.database[self.database["generation"] == generation]["chromosome"]
        else:
            fitness_value = None
        return fitness_pop.to_numpy(), pop.to_numpy()

    def update_no_write(self, chromosome:list, fitness_value:float, feasible:int)->None: 
        self.fitness_to_write.append(fitness_value)
        self.chromosome_to_write.append(chromosome)
        self.feasible_to_write.append(feasible)
        self.time_to_write.append(datetime.datetime.timestamp(datetime.datetime.now()))
    
    def update(
        self,
        chromosome: list,
        fitness_value: float,
        generation:int
    ) -> None:
       with open(self.name_database_with_ext, 'a') as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            timestamp = datetime.datetime.timestamp(datetime.datetime.now())
            df = self.get_empty_dataframe()
            df.loc[len(self.database)] = [timestamp, ','.join([str(x) for x in chromosome]), fitness_value, generation]
            df.to_csv(self.name_database_with_ext, index=False, mode="a", header=False)
            fcntl.flock(file, fcntl.LOCK_UN)
    @staticmethod
    def get_empty_dataframe():
        df = pd.DataFrame(
            columns=[
                "timestamp",
                "chromosome",
                "fitnessValue",
                "generation"
            ]
        )
        return df

    def rename(self, new_name: str) -> None:
        os.rename(self.name_database_with_ext, new_name + ".csv")
        self.name_database_with_ext = new_name + ".csv"