import psycopg2
import time


def connect(params_dic):
    """ Connect to the PostgreSQL database server """
    conn = None
    # connect to the PostgreSQL server
    print(' Connecting to the PostgreSQL database...')
    conn = psycopg2.connect(**params_dic)
    print(' Connection successful')

    return conn

def execute_many(conn, df, table):
    """
    Using cursor.executemany() to insert the dataframe
    """
    # Create a list of tupples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ','.join(list(df.columns))
    # SQL quert to execute
    query  = "INSERT INTO %s(%s) VALUES(%%s,%%s,%%s)" % (table, cols)
    cursor = conn.cursor()
    try:
        cursor.executemany(query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print(" execute_many() done")
    cursor.close()

if __name__ == "__main__":

    dataset = dataset

    param_dic = {
        "host"      : "#.#.#.#",
        "database"  : setting['dbname'],
        "user"      : setting['user'],
        "password"  : setting['password'],
        "port"      : "####"
    }

    conn = connect(param_dic)
    #execute_many(conn, dataset, 'etc_1')
    print(" execute_many() done")
