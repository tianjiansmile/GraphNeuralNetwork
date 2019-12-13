import pandas as pd

def read_edge_file(file_name):
    with open('../data/my/'+file_name+'edgelist.txt','r') as rf:
        with open('../data/my/'+file_name+'_label.txt','w') as wf:
            for line in rf.readlines():
                if line and line.strip() is not '':
                    # line.replace('\n','')
                    tmp = line.split('\t')
                    print(tmp)
                    id = tmp[0]
                    phone = tmp[1]
                    t = tmp[2]
                    wf.write(id+' 0'+'\n')
                    # if t == 'has_phone\n':
                    #     wf.write(phone+' 2'+'\n')
                    # else:
                    #     wf.write(phone +' 1'+'\n')


def data_merge():
    df = pd.read_csv('../data/emer/45456803word_vec.txt',sep=' ')
    df_label = pd.read_csv('../data/emer/45456803_label.txt',sep=' ')

    all_df = pd.merge(df, df_label, on='md5_num', how='inner')
    all_df = all_df.drop_duplicates(subset=['md5_num'],keep='first')

    print(all_df.shape)
    all_df.to_csv('../data/emer/45456803.content',index=False)

def map_label(x):
    if x:
        return 1


def data_merge1(edge_file,data_file):
    df = pd.read_csv('../data/'+data_file+'/'+edge_file+'word_vec.txt',sep=' ')
    df_label = pd.read_csv('../data/'+data_file+'/'+edge_file+'_label.txt', sep=' ')
    # all_df = df[['md5_num','v1']]

    df['v2'] = df['v1'].map(map_label)

    all_df = df[['md5_num','v2']]

    all_df = pd.merge(all_df, df_label, on='md5_num', how='inner')
    all_df = all_df.drop_duplicates(subset=['md5_num'], keep='first')

    print(all_df.shape)
    all_df.to_csv('../data/'+data_file+'/'+edge_file+'.content',index=False)

if __name__ == '__main__':
    edge_file = '2334530'
    data_file = 'user_network'
    # read_edge_file(edge_file)

    data_merge1(edge_file,data_file)