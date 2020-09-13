import pandas as pd


if __name__ == "__main__":




    vertebra_list = ['L1', 'L2', 'L3', 'L4', 'L5']
    disc_list = ['T12-L1', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']


    csv_path = r"D:\project\zjx\competitions\spark\code\model\axial_info_all.csv"

    all_csv = pd.read_csv(csv_path)


    print("vertebra:")
    vertebra_value = all_csv.loc[all_csv['identification'].isin(vertebra_list),'label'].value_counts()

    vertebra_weights_v1 =  2 * vertebra_value['v1']/vertebra_value.sum()
    vertebra_weights_v2 =  2 * vertebra_value['v2']/vertebra_value.sum()


    print(vertebra_value)
    # print(vertebra_weights_v1, vertebra_weights_v2)
    print("vertebra_weights:")

    vertebra_weights_list = [1/vertebra_weights_v1,1/vertebra_weights_v2]

    print(vertebra_weights_list)

    print("disc:")
    disc_value = all_csv.loc[all_csv['identification'].isin(disc_list),'label'].value_counts()
    print(disc_value)

    disc_weights_v1 = 5 * disc_value['v1']/vertebra_value.sum()
    disc_weights_v2 = 5 * (disc_value['v2'] + disc_value['v2,v5'])/vertebra_value.sum()
    disc_weights_v3 = 5 * (disc_value['v3'] + disc_value['v3'])/vertebra_value.sum()
    disc_weights_v4 = 5 * (disc_value['v4'] + disc_value['v4,v5'])/vertebra_value.sum()
    disc_weights_v5 = 5 * (disc_value['v5'] + disc_value['v5,v3'] + disc_value['v5,v2'])/vertebra_value.sum()


    disc_weights_list = [1/disc_weights_v1,1/disc_weights_v2,1/disc_weights_v3,1/disc_weights_v4,1/disc_weights_v5]
    print("disc_weiths:")
    print(disc_weights_list)

    # print(all_csv.loc[all_csv['identification'].isin(disc_list),'label'].value_counts())






