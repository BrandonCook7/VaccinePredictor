import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def graph(x,y, title):
    fig = px.scatter(x=x, y=y)
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'y': 1,
            'xanchor': 'center',
            'yanchor': 'top',
            'pad': {
                't': 25,
            }
        }
    )
    fig.show()

def graph_regression(x,y,regression_eq, state):
    df = pd.DataFrame({'Date': x, 'People Fully Vaccinated': y})
    fig = px.scatter(df, x="Date", y="People Fully Vaccinated")
    fig.update_layout(
      title={
        'text': f'Number of People Fully Vacinated in {state}',
        'x': 0.5,
        'y': 1,
        'xanchor': 'center',
        'yanchor': 'top',
        'pad': {
          't': 25,
        },
      },
      )
    
    fig.add_trace(go.Scatter(x=x, y=regression_eq, mode='lines', name='Prediction'))
    fig.show()

# def graph_vaccinated(date_until, state):
#     start_date = date(2021, 1, 12)
#     end_date = date_until
#     date_range = pd.date_range(start_date, end_date)
#     vaccination_percents = []
#     date_list = []
#     for index_date in date_range:
#         x = get_percent_vaccinated(index_date.strftime("%Y-%m-%d"), state)
#         vaccination_percents.append(x)
#         date_list.append(index_date.strftime("%Y-%m-%d"))
#     plt.plot(vaccination_percents)
#     plt.show()