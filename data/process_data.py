import sys, re
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load Messages Data function
    Arguments:
        messages_filepath -> messages file path CSV
        categories_filepath -> categories file path CSV
    Output:
        df -> return murge DataFrame
    """
    categories_df = pd.read_csv(categories_filepath)
    messages_df = pd.read_csv(messages_filepath)
    df = messages_df.merge(categories_df, on='id')
    return df

def clean_data(df):
    """
    Clean DataFrame function
    Arguments:
        df -> Dataframe to clean
    Output:
        df -> return clean dataframe
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.str.split('-').str[0]
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str.split('-').str[1]
        categories[column] = categories[column].astype(int)

    df = pd.concat([df, categories], axis=1, join='inner')
    # Drop the categories, as we converted values in the category dataframe
    df = df.drop(['categories'], axis=1)
    # Fix the value issue and converted to the 0 or 1
    df["related"] = df["related"].map(lambda x: 1 if x == 2 else x)
    # Remove duplicate records
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    """
    Save data to database
    Arguments:
        df -> Input dataframe
        filename -> name of the database file
    """
    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    df.to_sql(database_filename, engine, index=False)

def main():
    """
    main function
    Aeguments:
        messages_filepath -> CSV path of messages file
        categories_filepath -> CSV path of categories file
        database_filepath -> Name of the database file
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
