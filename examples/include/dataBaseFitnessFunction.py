import pandas as pd
import os
import datetime
import numpy as np 

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
        for idx,item in enumerate(self.database["chromosome"]):
            # Remove the square brackets
            cleaned_string = item.strip('[]')
            # Split the string by commas to get a list of strings
            string_list = cleaned_string.split(',')
            # Convert the list of strings to a list of floats
            float_list = [float(x) for x in string_list]
            if(float_list == chromosome): 
                fitness_value_1 = self.database["fitnessValue_1"][idx]
                fitness_value_2 = self.database["fitnessValue_2"][idx]
                return float(fitness_value_1), float(fitness_value_2)
        fitness_value_1 = float('nan')
        fitness_value_2 = float('nan')
        return fitness_value_1, fitness_value_2

    def update(
        self,
        chromosome: list,
        fitness_value_1: float,
        fitness_value_2: float,
    ) -> None:
        timestamp = datetime.datetime.timestamp(datetime.datetime.now())
        df = self.get_empty_dataframe()
        df.loc[len(self.database)] = [timestamp, str(chromosome), fitness_value_1, fitness_value_2]
        df.to_csv(self.name_database_with_ext, index=False, mode="a", header=False)

    @staticmethod
    def get_empty_dataframe():
        df = pd.DataFrame(
            columns=[
                "timestamp",
                "chromosome",
                "fitnessValue_1",
                "fitnessValue_2"
            ]
        )
        return df

    def rename(self, new_name: str) -> None:
        os.rename(self.name_database_with_ext, new_name + ".csv")
        self.name_database_with_ext = new_name + ".csv"
