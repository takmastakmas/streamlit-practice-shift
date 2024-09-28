import streamlit as st
import pandas as pd
import pulp
import japanize_matplotlib

class ShiftScheduler():

    def __init__(self):
        # リスト
        self.S = [] # スタッフのリスト
        self.D = [] # 日付のリスト
        self.SD = [] # スタッフと日付の組のリスト
        
        # 定数
        self.S2leader_flag = {} # スタッフの責任者フラグ
        self.S2min_shift = {} # スタッフの希望最小出勤日数
        self.S2max_shift = {} # スタッフの希望最大出勤日数
        self.D2required_staff = {} # 各日の必要人数
        self.D2required_leader = {} # 各日の必要責任者数
        
        # 変数
        self.x = {} # 各スタッフが各日にシフトに入るか否かを表す変数
        self.y_under = {} # 各スタッフの希望勤務日数の不足数を表すスラック変数
        self.y_over = {} # 各スタッフの希望勤務日数の超過数を表すスラック変数

        # 数理モデル
        self.model = None

        # 最適化結果
        self.status = -1 # 最適化結果のステータス
        self.sch_df = None # シフト表を表すデータフレーム
        
        
    def set_data(self, staff_df, calendar_df, penalty_a, ngdate):
        # リストの設定
        self.S = staff_df["スタッフID"].tolist()
        self.D = calendar_df["日付"].tolist()
        self.SD = [(s, d) for s in self.S for d in self.D]
        
        # 定数の設定
        S2Dic = staff_df.set_index("スタッフID").to_dict()
        self.S2leader_flag = S2Dic["責任者フラグ"]
        self.S2min_shift = S2Dic["希望最小出勤日数"]
        self.S2max_shift = S2Dic["希望最大出勤日数"]
        
        D2Dic = calendar_df.set_index("日付").to_dict()
        self.D2required_staff = D2Dic["出勤人数"]
        self.D2required_leader = D2Dic["責任者人数"]
        
        self.penalty_a = penalty_a
        self.S2ng_date = ngdate


    def show(self):
        print('='*50)
        print('Staffs:', self.S)
        print('Dates:', self.D)
        print('Staff-Date Pairs:', self.SD)

        print('Staff Leader Flag:', self.S2leader_flag)
        print('Staff Max Shift:', self.S2max_shift)
        print('Staff Min Shift:', self.S2min_shift)

        print('Date Required Staff:', self.D2required_staff)
        print('Date Required Leader:', self.D2required_leader)
        print('='*50)
        
        
    def build_model(self):
        ### 数理モデルの定義 ###
        self.model = pulp.LpProblem("ShiftScheduler", pulp.LpMinimize)
        
        ### 変数の定義 ###
        # 各スタッフの各日に対して、シフトに入るなら1、シフトに入らないなら0
        self.x = pulp.LpVariable.dicts("x", self.SD, cat="Binary")
        
        # 各スタッフの勤務希望日数の不足数を表すためのスラック変数
        self.y_under = pulp.LpVariable.dicts("y_under", self.S, cat="Continuous", lowBound=0)
        
        # 各スタッフの勤務希望日数の超過数を表すためのスラック変数
        self.y_over = pulp.LpVariable.dicts("y_over", self.S, cat="Continuous", lowBound=0)        

        ### 制約式の定義 ###
        # 各日に対して、必要な人数がシフトに入る
        for d in self.D:
            self.model += pulp.lpSum(self.x[s, d] for s in self.S) >= self.D2required_staff[d]
            
        # 各日に対して、必要なリーダーの人数がシフトに入る
        for d in self.D:
            self.model += pulp.lpSum(self.x[s, d] * self.S2leader_flag[s] for s in self.S) >= self.D2required_leader[d]
        
        # 希望休暇の制約
        for s in self.S:
            if self.S2ng_date[s] != "すべてOK":
                for d in self.D:
                    if d == self.S2ng_date[s]:
                        self.model += self.x[s, d] == 0
        
        ### 目的関数とスラック変数の定義 ###
        # 各スタッフの勤務希望日数の不足数と超過数を最小化する
        # self.model += pulp.lpSum([self.y_under[s] for s in self.S]) + pulp.lpSum([self.y_over[s] for s in self.S])
        self.model += pulp.lpSum([self.penalty_a[s] * self.y_under[s] for s in self.S]) + pulp.lpSum([self.penalty_a[s] * self.y_over[s] for s in self.S])
        
        # 各スタッフに対して、y_under[s]は勤務希望日数の不足数を表す
        for s in self.S:
            self.model += self.S2min_shift[s] - pulp.lpSum(self.x[s,d] for d in self.D) <= self.y_under[s]

        # 各スタッフに対して、y_over[s]は勤務希望日数の超過数を表す
        for s in self.S:
            self.model += pulp.lpSum(self.x[s,d] for d in self.D) - self.S2max_shift[s] <= self.y_over[s]

        
    def solve(self):
        solver = pulp.PULP_CBC_CMD(msg=0)
        self.status = self.model.solve(solver)

        print('status:', pulp.LpStatus[self.status])
        print('objective:', self.model.objective.value())

        Rows = [[int(self.x[s,d].value()) for d in self.D] for s in self.S]
        self.sch_df = pd.DataFrame(Rows, index=self.S, columns=self.D)

# タイトル
st.title("シフトスケジューリングアプリ")

# サイドバー
st.sidebar.header("データのアップロード")

st.sidebar.subheader('カレンダー')
uploaded_cal_file = st.sidebar.file_uploader("Browse files", key="calendar_file")

st.sidebar.subheader('スタッフ')
uploaded_st_file = st.sidebar.file_uploader("Browse files", key="staff_file")

# タブ
tab1, tab2, tab3 = st.tabs(["カレンダー情報", "スタッフ情報", "シフト表作成"])

with tab1:
    if uploaded_cal_file is not None:
      calendar_df = pd.read_csv(uploaded_cal_file)
      st.markdown("## カレンダー情報")
      st.write(calendar_df)
    else:
      st.info('カレンダー情報をアップロードしてください')

with tab2:
    if uploaded_st_file is not None:
        staff_df = pd.read_csv(uploaded_st_file)
        st.markdown("## スタッフ情報")
        st.write(staff_df)
        st.markdown("## 休暇希望")
        staff_list = staff_df['スタッフID'].tolist()
        ngdate = {}
        for staff in staff_list:
            st.write(f"スタッフID: {staff}")
            ngdate[staff] = st.radio(
                staff,
                ["すべてOK"] + [
                    calendar_df.loc[j, "日付"]
                    for j in range(calendar_df.shape[0])
                ],
                horizontal=True,
            )
    else:
      st.info('スタッフ情報をアップロードしてください')

with tab3:
    if uploaded_st_file is not None and uploaded_cal_file is not None:
        staff_list = staff_df['スタッフID'].tolist()
        penalty_a = {}
        for staff in staff_list:
            st.write(f"スタッフID: {staff}")
            penalty_a[staff] = st.slider(f"スタッフ{staff}の希望違反ペナルティ", 0, 100, 50, key=f"penalty_{staff}")
            
        if st.button('最適化実行'):
            st.write('シフト表を作成しました')
            st.markdown("## 最適化結果")
            shift_sch = ShiftScheduler()
            shift_sch.set_data(staff_df, calendar_df, penalty_a, ngdate)
            shift_sch.build_model()
            shift_sch.solve()
            st.write('実行ステータス:', pulp.LpStatus[shift_sch.status])
            st.write('最適値:', shift_sch.model.objective.value())
            
            st.markdown("## シフト表")
            st.write(shift_sch.sch_df)
            
            # ここにダウンロードボタンを追加
            csv = shift_sch.sch_df.to_csv().encode('utf-8')
            st.download_button(
                label="シフト表をダウンロード",
                data=csv,
                file_name='shift_schedule.csv',
                mime='text/csv',
            )
            
            st.markdown("## シフト数の充足確認")
            # 各日のシフト合計を取得
            day_totals = shift_sch.sch_df.sum(axis=0)
            # バーグラフを表示
            st.bar_chart(day_totals)

            st.markdown("## スタッフの希望の確認")
            # 各スタッフのシフト回数の合計を取得
            shift_totals = shift_sch.sch_df.sum(axis=1)
            # バーグラフを表示
            st.bar_chart(shift_totals)

            st.markdown("## 責任者の合計シフト数の充足確認")
            # 1. スタッフの「責任者フラグ」が1であるスタッフのIDを抽出
            leaders = staff_df[staff_df['責任者フラグ'] == 1]['スタッフID'].tolist()
            # 2. シフト表から、責任者のみのデータフレームを抽出
            leader_shifts = shift_sch.sch_df.loc[leaders]
            # 3. 各日の責任者数を集計
            daily_leader_count = leader_shifts.sum(axis=0)
            # 4. 結果を表示
            st.bar_chart(pd.DataFrame(daily_leader_count))
            
        else:
            st.write('シフト表を作成しましょう')
    else:
        st.write('カレンダー情報とスタッフ情報の両方をアップロードしてください')