from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import io
import pytest

from dask.dataframe.io.sql import read_sql_table
from dask.utils import tmpfile
from dask.dataframe.utils import assert_eq
pd = pytest.importorskip('pandas')
dd = pytest.importorskip('dask.dataframe')
pytest.importorskip('sqlalchemy')
pytest.importorskip('sqlite3')

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey

Base = declarative_base()


class Table(Base):
    __tablename__ = 'test'
    
    name = Column(String(32))
    number = Column(Integer, primary_key=True)
    age = Column(Integer)
    negish = Column(Integer)


class Table2(Base):
    __tablename__ = 'test2'
    
    id = Column(Integer, primary_key=True)
    job = Column(String(32))
    user_id = Column(Integer, ForeignKey('test.number'))


data = """
name,number,age,negish
Alice,0,33,-5
Bob,1,40,-3
Chris,2,22,3
Dora,3,16,5
Edith,4,53,0
Francis,5,30,0
Garreth,6,20,0
"""

data2 = """
id,job,user_id
0, Attacker, 0
1, Clerk, 1
2, Carter, 2
3, Explorator, 3
4, Piaf, 4
5, Bacon, 5
6, Blank, 6
"""

df = pd.read_csv(io.StringIO(data), index_col='number')
df2 = pd.read_csv(io.StringIO(data2), index_col='id')

df_join = df2.merge(df.reset_index(), right_on='number', left_on='user_id')
df_join = df_join[['name', 'number', 'job', 'age', 'negish']]
df_join.set_index('number', inplace=True)


def pytest_generate_tests(metafunc):
    if 'db' in metafunc.fixturenames:
        metafunc.parametrize("db", ['sqlite', 'mysql'], indirect=True)

@pytest.yield_fixture
def db(request):
    if request.param == 'sqlite':
        with tmpfile() as f:
            uri = 'sqlite:///%s' % f
            df.to_sql('test', uri, index=True, if_exists='replace')
            df2.to_sql('test2', uri, index=True, if_exists='replace')
            yield uri
            
    elif request.param == 'mysql':
        from sqlalchemy import create_engine
        uri = 'mysql+pymysql://x:x@x:3306/test'
        engine = create_engine(uri)

        from sqlalchemy_utils import database_exists, create_database, drop_database
        if database_exists(engine.url):
            drop_database(engine.url)
        create_database(engine.url)

        Base.metadata.create_all(engine)

        df.to_sql('test', uri, index=True, if_exists='append')
        df2.to_sql('test2', uri, index=True, if_exists='append')
        yield uri
        
        drop_database(engine.url)


def test_sa_selectable_1(db):
    from sqlalchemy.orm.query import Query

    q = Query([Table.name, Table.number, Table.age, Table.negish])
    data = read_sql_table(q.statement, db, columns=None, npartitions=2,
                          index_col='number').compute()
    
    assert (data.name == df.name).all()
    assert data.index.name == 'number'
    assert_eq(data, df)


def test_sa_selectable_2(db):
    from sqlalchemy.orm.query import Query

    q = Query([Table.name, Table.number, Table2.job, Table.age, Table.negish])\
        .join(Table2)
    data = read_sql_table(q.statement, db, columns=None, npartitions=2,
                          index_col='number').compute()
    
    assert (data.name == df_join.name).all()
    assert data.index.name == 'number'
    assert_eq(data, df_join)


def test_simple(db):
    # single chunk
    data = read_sql_table('test', db, npartitions=2, index_col='number'
                          ).compute()
    assert (data.name == df.name).all()
    assert data.index.name == 'number'
    assert_eq(data, df)


def test_npartitions(db):
    data = read_sql_table('test', db, columns=list(df.columns), npartitions=2,
                          index_col='number')
    assert len(data.divisions) == 3
    assert (data.name.compute() == df.name).all()
    data = read_sql_table('test', db, columns=['name'], npartitions=6,
                          index_col="number")
    assert_eq(data, df[['name']])
    data = read_sql_table('test', db, columns=list(df.columns),
                          bytes_per_chunk=2**30,
                          index_col='number')
    assert data.npartitions == 1
    assert (data.name.compute() == df.name).all()


def test_divisions(db):
    data = read_sql_table('test', db, columns=['name'], divisions=[0, 2, 4],
                          index_col="number")
    assert data.divisions == (0, 2, 4)
    assert data.index.max().compute() == 4
    assert_eq(data, df[['name']][df.index <= 4])


def test_division_or_partition(db):
    with pytest.raises(TypeError):
        read_sql_table('test', db, columns=['name'], index_col="number",
                       divisions=[0, 2, 4], npartitions=3)

    out = read_sql_table('test', db, index_col="number", bytes_per_chunk=100)
    m = out.map_partitions(lambda d: d.memory_usage(
        deep=True, index=True).sum()).compute()
    assert (50 < m).all() and (m < 200).all()
    assert_eq(out, df)


def test_range(db):
    data = read_sql_table('test', db, npartitions=2, index_col='number',
                          limits=[1, 4])
    assert data.index.min().compute() == 1
    assert data.index.max().compute() == 4


def test_datetimes():
    import datetime
    now = datetime.datetime.now()
    d = datetime.timedelta(seconds=1)
    df = pd.DataFrame({'a': list('ghjkl'), 'b': [now + i * d
                                                 for i in range(2, -3, -1)]})
    with tmpfile() as f:
        uri = 'sqlite:///%s' % f
        df.to_sql('test', uri, index=False, if_exists='replace')
        data = read_sql_table('test', uri, npartitions=2, index_col='b')
        assert data.index.dtype.kind == "M"
        assert data.divisions[0] == df.b.min()
        df2 = df.set_index('b')
        assert_eq(data.map_partitions(lambda x: x.sort_index()),
                  df2.sort_index())


def test_with_func(db):
    from sqlalchemy import sql
    index = sql.func.abs(sql.column('negish')).label('abs')

    # function for the index, get all columns
    data = read_sql_table('test', db, npartitions=2, index_col=index)
    assert data.divisions[0] == 0
    part = data.get_partition(0).compute()
    assert (part.index == 0).all()

    # now an arith op for one column too; it's name will be 'age'
    data = read_sql_table('test', db, npartitions=2, index_col=index,
                          columns=[index, -sql.column('age')])
    assert (data.age.compute() < 0).all()

    # a column that would have no name, give it a label
    index = (-sql.column('negish')).label('index')
    data = read_sql_table('test', db, npartitions=2, index_col=index,
                          columns=['negish', 'age'])
    d = data.compute()
    assert (-d.index == d['negish']).all()


def test_no_nameless_index(db):
    from sqlalchemy import sql
    index = (-sql.column('negish'))
    with pytest.raises(ValueError):
        read_sql_table('test', db, npartitions=2, index_col=index,
                       columns=['negish', 'age', index])

    index = sql.func.abs(sql.column('negish'))

    # function for the index, get all columns
    with pytest.raises(ValueError):
        read_sql_table('test', db, npartitions=2, index_col=index)


def test_select_from_select(db):
    from sqlalchemy import sql
    s1 = sql.select([sql.column('number'), sql.column('name')]
                    ).select_from(sql.table('test'))
    out = read_sql_table(s1, db, npartitions=2, index_col='number')
    assert_eq(out, df[['name']])
