### Amirali Bagherzadeh
### 9912743386

import numpy as np
import pandas as pd
import seaborn as sns
from mplsoccer import Pitch, VerticalPitch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec


events = pd.read_pickle('C:\\Users\\USER\\Desktop\\Term 7\\CIS\\P4\\2018_FIFA_WORLD_CUP_EVENTS.pkl')
matches = pd.read_pickle('C:\\Users\\USER\\Desktop\\Term 7\\CIS\\P4\\2018_FIFA_WORLD_CUP_MATCHES.pkl')


def continent(country_name): 
    ASIA = ['Iran', 'South Korea', 'Australia', 'Japan', 'Saudi Arabia']
    EUROPE = ['England', 'Sweden', 'Portugal','Croatia', 'Serbia', 'Belgium', 'Poland', 'France', 'Germany', 'Switzerland', 'Denmark', 'Iceland', 'Spain', 'Russia']
    NORTH_AMERICA = ['Panama', 'Costa Rica', 'Mexico']
    SOUTH_AMERICA = ['Colombia', 'Brazil', 'Uruguay', 'Argentina', 'Peru']
    AFRICA = ['Egypt', 'Tunisia', 'Nigeria', 'Senegal', 'Morocco']
    if country_name in ASIA:
        return 'Asia'  
    elif country_name in EUROPE:
       return 'Europe' 
    elif country_name in NORTH_AMERICA:
       return 'North_America'
    elif country_name in SOUTH_AMERICA:
       return 'South_America'
    else:
       return 'Africa'
    
def format_negative(value, _):
    return abs(value)

def event_count(df,list, a):
    for i in range(len(df)):
        if df.iloc[i,5] == 'Offside':
            list[0] += a
        elif df.iloc[i,5] == 'Dispossessed':
            list[1] += a
        elif df.iloc[i,5] == 'Miscontrol':
            list[2] += a
        elif df.iloc[i,5] == 'Error':
            list[3] += a
        elif df.iloc[i,5] == 'Foul Committed':
            list[4] += a
        elif df.iloc[i,5] == 'Dribbled Past':
            list[5] += a

def timing(minute, period):
    if period == 3:
        return '1st_ext'
    elif period == 4:
        return '2nd_ext'
    elif period == 1:
        if minute < 15:
            return '0-15'
        elif 15 <= minute < 30:
            return '15-30'
        elif minute >= 30:
            return '30-1st_half'
    elif period == 2:
        if minute < 60:
            return '45-60'
        elif 60 <= minute < 75:
            return '60-75'
        elif minute >= 75:
            return '75-2nd_half'

def extra_time(minute,second,period):
    if period == 1:
        return (minute - 45 + second / 60)
    if period == 2:
        return (minute - 90 + second / 60)
    if period == 3:
        return (minute - 105 + second / 60)
    if period == 4:
        return (minute - 120 + second / 60)

def euclidean_distance(loc1, loc2):
    return ((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)**0.5


matches['continent'] = matches['home_team'].apply(lambda x: continent(x))
matches['games'] = matches.apply(lambda row: set([row['home_team'], row['away_team']]), axis=1)

### 1-------------------------------------------------------------------------------------
def output_1(events, matches):
    Shots = events[(events.event_name == 'Shot') & (events.event_period != 5)]
    Shots.index = list(range(len(Shots)))
    Shots = pd.concat([Shots.iloc[:,:6],pd.DataFrame(list(Shots.event_details))], axis = 1)
    goals = Shots[Shots.outcome == 'Goal']
    goals = pd.concat([goals, events[(events.event_name == 'Own Goal For') & (events.event_period.isin([1, 2, 3, 4]))]], axis = 0)
    goals.index = list(range(len(goals)))
    goals['timing'] = [timing(goals.event_time[i][0],goals.event_period[i]) for i in range(len(goals))]

    plt.figure(figsize=(10, 5))
    order = ['0-15','15-30','30-1st_half','45-60','60-75','75-2nd_half','1st_ext','2nd_ext']
    sns.countplot(x= 'timing', data = goals, order = order, hue = 'timing', palette = 'viridis', legend = False)
    plt.xlabel('time period')
    plt.ylabel('Goals')
    plt.title('First output')
    plt.show()



### 2---------------------------------------------------------------------------------------
def output_2(events, matches):
    ext_time = events[(events.event_name == 'Half End') & (events.event_period != 5)]
    ext_time.loc[:, 'extra_time'] = ext_time.apply(lambda row: extra_time(row['event_time'][0], row['event_time'][1],row['event_period']), axis=1)
    ext_time.index = list(range(len(ext_time)))

    referee_info = matches[['referee','match_id']]
    merged_df = pd.merge(ext_time, referee_info, on = 'match_id', how = 'left')
    referee_ext_time = merged_df.groupby('referee')['extra_time'].sum().reset_index()
    referee_ext_time['extra_time'] = referee_ext_time['extra_time'] / 2
    referee_ext_time = referee_ext_time.sort_values(by='extra_time', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(y = referee_ext_time.referee, x = referee_ext_time.extra_time, hue=referee_ext_time.extra_time, orient='h')
    plt.xlabel('Ext-time')
    plt.ylabel('Referee')
    plt.title('Extra time by each Referee')
    plt.show()

### 3------------------------------------------------------------------------------
def output_3(events, matches):
    fouls = events[events.event_name == 'Foul Committed'].reset_index()
    fouls = pd.concat([fouls.iloc[:,:6],pd.DataFrame(list(fouls.event_details))], axis = 1)
    cards = fouls[fouls.card != 'None'].reset_index()
    date = matches[["match_date", "match_id"]]
    cards = pd.merge(cards, date, on = 'match_id', how='left')
    cards = cards.groupby(['match_date','card']).size().unstack()
    cards = cards.fillna(0)
    colors = {'red_card': 'red','second_yellow_card': 'orange','yellow_card': 'gold'}
    cards.plot(kind='line', marker='o', figsize=(10, 6), color = colors.values())
    plt.xlabel('time period')
    plt.ylabel('Goals')
    plt.title('First output')
    plt.legend(title='Card Type', labels=colors.keys())
    plt.show()

### 4--------------------------------------------------------------------------
def output_4(events, matches):
    passes = events[events.event_name == 'Pass'].reset_index()
    passes = pd.concat([passes.iloc[:,1:6],pd.DataFrame(list(passes.event_details))], axis = 1)
    passes = passes[(passes.outcome == 'complete') & (passes.recipient != 'None')]
    passes.index = list(range(len(passes)))
    passes['unique_players'] = passes.apply(lambda row: set([row['player'], row['recipient']]), axis=1)
    players_pass = passes['unique_players'].value_counts()
    top_10 = players_pass[:10].reset_index()
    top_10.index = list(range(1,len(top_10)+1))
    top_10.columns = ['players', 'counts']
    print(top_10)

    number = int(input("Which want do you want to see? "))
    players = top_10.players[number]
    pass_type = passes[passes.unique_players == players]['height'].value_counts()
    pass_in_game = passes[passes.unique_players == players][['location', 'end_location']]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    sns.barplot(x=top_10.counts, y=top_10.players.astype(str), hue=top_10.players.astype(str), palette="viridis", ax=axs[0])
    axs[0].set_xlabel('Number of passes')
    axs[0].set_ylabel('Players')
    axs[0].set_title('Top 10 in passes')

    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='grass', line_color='white', goal_type='box')
    pitch.draw(ax=axs[1])
    for i in range(len(pass_in_game)):
        axs[1].plot([pass_in_game.iloc[i, 0][1], pass_in_game.iloc[i, 1][1]],
                [pass_in_game.iloc[i, 0][0], pass_in_game.iloc[i, 1][0]],
                color='blue')
    axs[1].set_title('Passes Distribution')

    axs[2].pie(pass_type, labels=pass_type.index, autopct='%1.1f%%', colors=['pink', 'cyan', 'olive'])
    axs[2].set_title('Passes Height Proportion')

    plt.tight_layout()
    plt.show()



### 5 -----------------------------------------------------------
def output_5(events, matches):
    substitute = events[events.event_name == 'Substitution'].reset_index()
    substitute = pd.concat([substitute.iloc[:, 1:7], pd.DataFrame(list(substitute.event_details))], axis=1)
    substitute['unique_players'] = substitute.apply(lambda row: set([row['player'], row['replacement']]), axis=1)
    players_sub = substitute['unique_players'].value_counts()
    top_10_sub = players_sub[:10].reset_index()
    top_10_sub.index = list(range(1,len(top_10_sub)+1))
    top_10_sub.columns = ['players', 'count']
    print("======> The 10 Highest Number of Substitutions Between 2 Players : ")
    print(top_10_sub)

    number = int(input("Which of 2 Players Do you want to see more details about their substitution? "))
    players = top_10_sub.players[number]
    sub_detail = substitute[substitute.unique_players == players]

    home_manager = matches.loc[:, ['home_team', 'home_manager']].rename(columns={'home_team': 'team', 'home_manager': 'manager'})
    away_manager = matches.loc[:, ['away_team', 'away_manager']].rename(columns={'away_team': 'team', 'away_manager': 'manager'})
    managers = pd.concat([home_manager,away_manager],axis = 0)
    managers = managers.drop_duplicates()
    managers.index = list(range(len(managers)))

    matches['2 Teams'] = matches['home_team'] + ' vs ' + matches['away_team']
    sub_detail = pd.merge(sub_detail, matches[['2 Teams', 'match_id']], on = 'match_id', how='left')
    sub_detail = pd.merge(sub_detail, managers, on = 'team', how='left')

    sub_detail.loc[:, 'event_time'] = [f"{min}:{sec}" for min, sec in sub_detail['event_time']]
    report = sub_detail[['2 Teams','event_time', 'team', 'manager', 'player', 'replacement', 'reason']]
    print(report)

### 6------------------------------------------------------------
def output_6(events, matches):
    play_pattern = events[(events.event_name == 'Shot') & (events.event_period != 5)].reset_index()
    play_pattern = pd.concat([play_pattern.iloc[:,1:7],pd.DataFrame(list(play_pattern.event_details))], axis = 1)
    play_pattern = play_pattern[play_pattern.outcome == 'Goal']
    play_pattern.index = list(range(len(play_pattern)))
    play_pattern = pd.merge(play_pattern, matches[['continent', 'match_id']], on = 'match_id', how='left')

    table = pd.crosstab(play_pattern['continent'], play_pattern['play_pattern'])
    table.plot.bar(stacked=True, width=0.9)
    plt.title('Goals for each continent Stacked Bar Chart')
    plt.xlabel('Continent')
    plt.ylabel('Frequency')
    plt.legend(title='Play Pattern')
    plt.show()

# ### 7 -----------------------------------------------------------
def output_7(events, matches):
    country = input("Please enter Country Name : ")
    tactic = events[((events.team == country) & (events.event_name == 'Starting XI')) | ((events.team == country) & (events.event_name == 'Tactical Shift'))].reset_index()
    tactic = pd.concat([tactic.iloc[:,1:7],pd.DataFrame(list(tactic.event_details))], axis = 1)
    matches['2 Teams'] = matches['home_team'] + ' vs ' + matches['away_team']
    tactic = pd.merge(tactic, matches[['match_id','2 Teams']], on = 'match_id', how='left')
    tactic = pd.merge(tactic, matches[['match_id','2 Teams','match_date']], on = 'match_id', how='left')
    tactic = tactic.sort_values(by='match_date')    

    formation = []
    teams = tactic.iloc[0,7]
    for i in range(len(tactic)):
        if teams != tactic.iloc[i,7]:
            print('========> Match',tactic.iloc[i-1,0],teams)
            result = ' --> '.join(map(str, formation))
            print(result)
            print("==============================================")
            teams = tactic.iloc[i,7]
            formation.clear()
            formation.append(tactic.iloc[i,6])
        else:
            formation.append(tactic.iloc[i,6])
            
    print('Match',tactic.iloc[i,0],teams)
    result = ' --> '.join(map(str, formation))
    print(result)

## 8 ------------------------------------------------------------
def output_8(events, matches):
    comp_pass = events[(events.event_name == 'Pass')].reset_index()
    comp_pass = pd.concat([comp_pass.iloc[:,1:7],pd.DataFrame(list(comp_pass.event_details))], axis = 1)
    comp_pass = comp_pass[comp_pass.outcome == "complete"]

    comp_pass['length'] = comp_pass.apply(lambda x: euclidean_distance(x['location'], x['end_location']), axis=1)

    num_country = int(input("How many country do you want to prosecc ?"))
    country_list = []
    complete_pass = []
    avg_length = []
    for i in range(num_country):
        country = input("Please enter Country Name : ")
        country_list.append(country)
        count_result = comp_pass.loc[comp_pass['team'] == country, 'outcome'].count()
        complete_pass.append(count_result)
        mean_length = comp_pass.loc[comp_pass['team'] == country, 'length'].mean()
        avg_length.append(mean_length)

    print('Countries : ',' - '.join(country_list))
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.bar(country_list, avg_length, color='blue')
    plt.xlabel('Country')
    plt.ylabel('Average of correct passes length')

    plt.subplot(1,2,2)
    plt.bar(country_list, complete_pass, color='red')
    plt.xlabel('Country')
    plt.ylabel('Number of complete pass')

    plt.show()


### 9 ---------------------------------------------------------------
def output_9(events, matches):
    Shots = events[(events.event_name == 'Shot')].reset_index()
    Shots = pd.concat([Shots.iloc[:,1:7],pd.DataFrame(list(Shots.event_details))], axis = 1)
    matches['games'] = matches.apply(lambda row: set([row['home_team'], row['away_team']]), axis=1)

    country_1 = input("Please enter first Country : ")
    country_2 = input("Please enter second Country : ")
    match_id = list(matches['match_id'][matches.games == set([country_1,country_2])])

    Shots = Shots[(Shots.match_id.isin(match_id))]
    Shots['X'] = Shots['location'].apply(lambda x: x[0])
    Shots['Y'] = Shots['location'].apply(lambda x: x[1])

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(8,8)
    field = fig.add_subplot(gs[0:4, :])
    field.set_title("Football Pitch")
    pitch = VerticalPitch(pitch_color='grass', line_color='white', stripe = True,half=True)
    pitch.draw(ax=field)
    sns.scatterplot(data = Shots, x = 'Y', y = 'X', hue = 'team', size = 'xg', palette= ['blue', 'red'], sizes = (50,500),alpha = 0.8 ,ax = field)

    pie_1 = fig.add_subplot(gs[4:, :4])
    outcome_1 = Shots[Shots['team'] == country_1]
    outcome_1 = outcome_1['outcome'].value_counts()
    outcome_1.plot(kind='pie', autopct='%1.1f%%', ax=pie_1)
    pie_1.set_title(country_1)

    pie_2 = fig.add_subplot(gs[4:, 4:])
    outcome_2 = Shots[Shots['team'] == country_2]
    outcome_2 = outcome_2['outcome'].value_counts()
    outcome_2.plot(kind='pie', autopct='%1.1f%%', ax=pie_2)
    pie_2.set_title(country_2)

    plt.show()


### 10 ----------------------------------------------------------
def output_10(events, matches):
    country_1 = input("Please enter first Country : ")
    country_2 = input("Please enter second Country : ")
    matches['games'] = matches.apply(lambda row: set([row['home_team'], row['away_team']]), axis=1)
    match_id = list(matches['match_id'][matches.games == set([country_1,country_2])])

    eve_name = ['Offside','Dispossessed','Miscontrol','Error','Foul Committed','Dribbled Past'] 
    new_event_1 = events[(events.match_id.isin(match_id)) & (events.event_name.isin(eve_name)) & (events.team == country_1)]
    new_event_2 = events[(events.match_id.isin(match_id)) & (events.event_name.isin(eve_name)) & (events.team == country_2)]
    new_event_1.index = list(range(len(new_event_1)))
    new_event_2.index = list(range(len(new_event_2)))

    eve_num_1 = [0,0,0,0,0,0]
    eve_num_2 = [0,0,0,0,0,0]

    event_count(new_event_1, eve_num_1, 1)
    event_count(new_event_2, eve_num_2, -1)


    fig, ax = plt.subplots()
    ax.barh(eve_name, eve_num_1, color='olive', label = country_1, alpha=0.8)
    ax.barh(eve_name, eve_num_2, color='red', label = country_2, alpha = 0.8)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Events')
    ax.set_title('Tornado Graph')
    ax.xaxis.set_major_formatter(FuncFormatter(format_negative))
    ax.legend()

    plt.show()



while True:
    print("-----------WELLCOME TO THE WORLD CUP ANALYSIS--------------")
    print("CHOOSE FROM THE FOLLOWING : ")
    print("(A) TOURNAMENT REPORT")
    print("(B) TEAM REPORT")
    print("(C) MATCH REPORT")
    print("(D) EXIT")
    choice = input("PLEASE ENTER YOUR CHOICE : ").upper()
    if choice == 'A':
        while True:
            print("--------------TOURNAMENT REPORT--------------")
            print("CHOOSE FROM THE FOLLOWING : ")
            print("(1) NUMBER OF SCORED GOALS IN  EACH 15 MINUTES TIME INTERVALS [VERTICAL BAR CHART]")
            print("(2) SUM OF CALCULATED EXTRA TIMES BY EACH REFEREE IN THE TOURNAMENT [HORIZONTAL BAR CHART]")
            print("(3) TIME SERIES OF RED AND YELLOW CARDS [LINE GRAPH]")
            print("(4) TOP 10 OF HIGHEST NUMBER OF PASSES BETWEEN 2 PLAYERS [DATA FRAME & SUBPLOT(1,3)]")
            print("(5) TOP 10 OF HIGHEST NUMBER OF SUBSTITUATIONS BETWEEN 2 PLAYERS [DATA FRAME]")
            print("(6) NUMBER OF GOALS FOR EACH CONTINENT [STACKED BAR CHART]")
            print("(7) BACK")   
            choice = input("PLEASE ENTER YOUR CHOICE : ")
            if choice == '1':
                output_1(events, matches)
            elif choice == '2':
                output_2(events, matches)
            elif choice == '3':
                output_3(events, matches)
            elif choice == '4':
                output_4(events, matches)
            elif choice == '5':
                output_5(events, matches)
            elif choice == '6':
                output_6(events, matches)
            elif choice == '7':
                print("BACK TO THE MAIN MENU")
                break
            else:
                print("INVALID CHOICE")
                break
    elif choice == 'B':
        while True:
            print("--------------TEAM REPORT--------------")
            print("(1) TACTICAL SHIFT REPORT [DATA FRAME]")
            print("(2) AVERAGE LENGTH OF CORRECT PASSES & NUMBER OF COMPLETE PASSES [SUBPLOT (1,2)]")
            print("(3) BACK")
            choice = input("PLEASE ENTER YOUR CHOICE : ")
            if choice == '1':
                output_7(events, matches)
            elif choice == '2':
                output_8(events, matches)
            elif choice == '3':
                print("BACK TO THE MAIN MENU")
                break
            else:
                print("INVALID CHOICE")
                break
    elif choice == 'C':
        while True:
            print("--------------MATCH REPORT--------------")
            print("(1) LOCATION OF SHOTS DURING SPECEFIC MATCH FOR BOTH OPPONENTS [SUBPLOT]")
            print("(2) 6 FEAUTURES TORNADO PLOT FOR BOTH OPPONENTS [TORNADO]")
            print("(3) BACK")
            choice = input("PLEASE ENTER YOUR CHOICE : ")
            if choice == '1':
                output_9(events, matches)
            elif choice == '2':
                output_10(events, matches)
            elif choice == '3':
                print("BACK TO THE MAIN MENU")
                break
            else:
                print("INVALID CHOICE")
                break
    elif choice == 'D':
        print("THANKS FOR USING THE WORLD CUP ANALYSIS")
        break
    else :
        print("INVALID CHOICE")