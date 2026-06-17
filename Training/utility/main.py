from pathlib import Path
import pandas as pd
def read_data () :
    BASE_DIRECTORY = Path(__file__).parent
    data_csv_path = BASE_DIRECTORY.parent / "data" / "diabetic_data.csv"
    try : 
       df = pd.read_csv ( data_csv_path)
       return df 
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
   
    
