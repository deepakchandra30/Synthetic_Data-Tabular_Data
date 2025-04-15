import pandas as pd 
from sdv.single_table import TVAESynthesizer
from sdv.constraints import ScalarInequality
from sdv.metadata import Metadata
from table_evaluator import TableEvaluator

df1 = pd.read_csv('Dataset_v2.csv')

#Dropping the first three columns as we don't want synthetic values for them, we are gonna add them backt to the dataframe later
dropped_columns = df1[['Year', 'Regional Veterinary Offices', 'C03321V04008']]
df = df1.drop(columns=['Year', 'Regional Veterinary Offices', 'C03321V04008'])

#obtaining metadata from the dataframe
metadata = Metadata.detect_from_dataframe(data=df)

metadata.remove_primary_key()

metadata.update_column(
    column_name='DAA01C06',
    sdtype='categorical')

metadata.update_column(
    column_name='DAA01C06',
    sdtype='numerical')

metadata.update_column(
    column_name='DAA01C07',
    sdtype='numerical')

metadata.validate()

#print(metadata)
#metadata.save_to_json('metadata.json')

#Using builtin constraint to say the model that column 1 is always greater than column 2
Inequality_columns = {
    'constraint_class': 'Inequality',
    'constraint_parameters': {
        'low_column_name': 'DAA01C02',
        'high_column_name': 'DAA01C01',
        'strict_boundaries': True
    }
}

synthesizer = TVAESynthesizer(metadata, epochs= 625, verbose=True)

#Custom Constraints to set the second relationship
synthesizer.load_custom_constraint_classes(
    filepath ='Constraint1.py',
    class_names= ['Custom_constraint1']
)

myconstraint1 = {
    'constraint_class' : 'Custom_constraint1',
    'constraint_parameters' : {
        'column_names' :['DAA01C02','DAA01C03','DAA01C04']
    }
}

#adding the constraints to the model
synthesizer.add_constraints(constraints=[Inequality_columns, myconstraint1])

synthesizer.fit(data=df)
samples = synthesizer.sample(406)

#adding the dropped columns to the generated data
synthetic_data_with_dropped_columns = pd.concat([dropped_columns, samples], axis=1)
synthetic_data_with_dropped_columns.to_csv('Synthetic_data_TVAE.csv', index = False)


#Evaluation Code
print(df.shape, samples.shape)

evaluator = TableEvaluator(df,samples, cat_cols = None)
evaluator.visual_evaluation()
