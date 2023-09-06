import streamlit as st
import pandas as pd
from math import ceil
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import webbrowser

st.set_page_config(layout='wide')


@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


def clear_wishlist():
    st.session_state["wishlist"] = {}


def add_to_wishlist(plant_index):
    st.session_state["wishlist"][plant_index] = 1


def remove_from_wishlist(plant_index):
    if plant_index in st.session_state.wishlist:
        st.session_state["wishlist"].pop(plant_index)


def search_google(keyword):
    webbrowser.open_new_tab(f"https://www.google.com/search?q={keyword}")


def clear_history():
    st.session_state['chat_dialogue'] = [{"role": "system"}]


def render_grid(df, key=""):

    if not df.empty:

        controls = st.columns(3)
        with controls[0]:
            batch_size = st.select_slider("Batch size:",range(10,110,10), key=f"{key}_batch_size")
        with controls[1]:
            row_size = st.select_slider("Row size:", range(1,6), value = 4, key=f"{key}_row_size")
        num_batches = ceil(len(df)/batch_size)
        with controls[2]:
            page = st.selectbox("Page", range(1,num_batches+1), key=f"{key}_page")

        batch = df[(page-1)*batch_size : page*batch_size]
        st.dataframe(batch, use_container_width=True)
        
        grid = st.columns(row_size)
        col = 0

        for idx, plant in batch.to_dict(orient='index').items():

            with grid[col]:

                if key == "selected":
                    button_cols = st.columns([0.7, 0.2, 0.1])
                else:
                    button_cols = st.columns([0.8, 0.2])

                button_cols[0].button(
                    f"{plant['Scientific_Name']} - {plant['Plant_Name']} - {plant['Coverage']} sqft", 
                    key=f"{key}_google_{idx}",
                    on_click=search_google, 
                    args=[f"{plant['Scientific_Name']}"], 
                    type="primary", 
                    use_container_width=True
                )

                if key == "selected":
                    st.session_state.wishlist[idx] = button_cols[1].number_input("Quantity", key=f"{key}_wishlist_qty_{idx}", min_value=1, value=1, label_visibility="collapsed")
                    button_cols[2].button(
                        "🗑️",
                        key=f"{key}_cart_{idx}",
                        on_click=remove_from_wishlist,
                        args=[idx],
                        type="secondary",
                        use_container_width=True
                    )
                else:
                    button_cols[1].button(
                        "🛒",
                        key=f"{key}_cart_{idx}",
                        on_click=add_to_wishlist,
                        args=[idx],
                        type="secondary",
                        use_container_width=True
                    )

                for i, _ in plant['source'].items():
                    try:
                        st.image(plant['source'].get(i), use_column_width=True, caption=plant['source'].get(i))
                    except Exception as err:
                        st.write(f"☹️ error loading image due to {err}")

                st.divider()

            col = (col + 1) % row_size


def help_doc():

    with st.expander("🤔 How to use this app?"):
        st.info("This app is to help you navigate plant selection for you [Valley Water Landscape Rebate Program](https://valleywater.dropletportal.com/)")
        st.divider()
        st.image("instructions/main_screen.png", use_column_width=True)
        st.divider()
        st.image("instructions/selection.png", use_column_width=True)
        st.divider()
        st.image("instructions/chatbot.png", use_column_width=True)
        st.divider()



def render_chatbot():

    chat_model = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        temperature=0.7
    )

    # Display chat messages from history on app rerun
    for message in st.session_state.chat_dialogue_display:
        with st.chat_message(message["role"]):
            st.markdown(message["content"].content)

    selected = st.session_state.orig_df.iloc[list(st.session_state["wishlist"].keys())]

    template = f""" You are a helpful landscape professional who help select plants for my drought-proof garden project in California.
    Currently, the user have selected '{','.join(selected['Scientific_Name'])}' as a combination. You should help provide 
    helpful guidance upon asked. While answering questions, you want to comprehensively consider the garden design, plant combination, 
    planting tips, overall budget for a successful garden project at northern california.
    """

    if template != st.session_state.chat_dialogue[0].get('content', ''):  # Reset dialogue when user updates the cart
        st.session_state.chat_dialogue = [{"role": "system", "content": SystemMessage(content=template)}]

    if prompt := st.chat_input("Type your response"):

        st.session_state.chat_dialogue.append({"role": "user", "content": HumanMessage(content=prompt)})
        st.session_state.chat_dialogue_display.append({"role": "user", "content": HumanMessage(content=prompt)})

        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response = chat_model([i["content"] for i in st.session_state.chat_dialogue])
            message_placeholder.markdown(response.content + "▌")
        
        st.session_state.chat_dialogue.append({"role": "assistant", "gcontent": AIMessage(content=response.content)})
        st.session_state.chat_dialogue_display.append({"role": "assistant", "content": AIMessage(content=response.content)})


def render_app():

    st.title("🌿🌵Valley Water Rebate Plants View💐🌾")

    help_doc()

    if 'wishlist' not in st.session_state:
        st.session_state['wishlist'] = {}
    
    if 'chat_dialogue' not in st.session_state:
        st.session_state['chat_dialogue'] = [{"role": "system"}]

    if 'chat_dialogue_display' not in st.session_state:
        st.session_state['chat_dialogue_display'] = []

    df = pd.read_json("Valley_Water_Qualified_Plants_sourced.json")
    df = df.reset_index(drop=True)

    st.session_state['orig_df'] = df.copy()

    binary_cols = {
        'Bamboo': False, 
        'Bulb': True, 
        'Grass': True, 
        'Groundcover': True, 
        'Perennial': True, 
        'Palm': True, 
        'Shrub': False, 
        'Succulent': True, 
        'Tree': False, 
        'Vine': False, 
        'Native': True, 
        'Genetic_Concerns': False, 
        'Potentially_Invasive': False,
    }

    user_filter = {}

    # Sidebar Filters
    st.sidebar.header("Filters")
    search_str = st.sidebar.selectbox("Search by Name", options=[""] + df['Scientific_Name'].unique().tolist() + df['Plant_Name'].unique().tolist(), index=0, placeholder="")
    (coverage_min, coverage_max) = st.sidebar.slider(
        "Coverage (sqft)", 
        min_value=0, 
        max_value=int(df.Coverage.max()), 
        value=(0, int(df.Coverage.max()))
    )
    sidebar_cols = st.sidebar.columns(2)
    for ind, (k, default) in enumerate(binary_cols.items()):
        user_filter[k] = sidebar_cols[ind % 2].checkbox(k, value=default)

    if len(search_str) == 0:
        for k, val in user_filter.items():
            if not val:
                df = df[df[k] == 'No']

        df = df[df.Coverage > 0]
        df = df[(df.Coverage >= coverage_min) & (df.Coverage <= coverage_max)]
    else:
        df = df[df['Scientific_Name'].str.contains(search_str) | df['Plant_Name'].str.contains(search_str)]

    df = df.sort_values('Scientific_Name')

    # Main Selection Screen
    tab_available, tab_selected = st.tabs(["🌿 Available Plant", "🛒 Selected Plant"])

    with tab_available:
        st.subheader(f"{len(df)} Plants Available")
        render_grid(df[~df.index.isin(list(st.session_state["wishlist"].keys()))], key="available")
        
    with tab_selected:
        if len(st.session_state["wishlist"].keys()) == 0:
            st.subheader("Add plants to your wish list from Available Plant 🌿 Tab")
        else:
            render_grid(st.session_state.orig_df.iloc[list(st.session_state["wishlist"].keys())], key="selected")


    # Wishlist Controls
    st.sidebar.header("Your Wish List")

    if len(st.session_state["wishlist"]) > 0:
        selected = st.session_state.orig_df.iloc[list(st.session_state["wishlist"].keys())]
        total_coverage = 0
        for idx, plant in selected.to_dict(orient='index').items():
            total_coverage += st.session_state.wishlist[idx] * int(plant["Coverage"])
    else:
        selected = {}
        total_coverage = 0
    
    st.sidebar.info(f"Selected {len(selected)} Plants | Total Coverage is {total_coverage}")

    st.sidebar.button("🤖 Clear Chat History", use_container_width=True, on_click=clear_history)
    st.sidebar.button("🫙 Clear Wish List", use_container_width=True, on_click=clear_wishlist)
    st.sidebar.download_button(
        "🗳️ Download Wish List",
        convert_df(st.session_state.orig_df.iloc[list(st.session_state["wishlist"].keys())]),
        "wishlist.csv",
        "text/csv",
        key='download-csv',
        use_container_width=True
    )

    render_chatbot()


render_app()