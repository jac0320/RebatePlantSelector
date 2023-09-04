import streamlit as st
import pandas as pd
from math import ceil
import os
from copy import deepcopy
import webbrowser

st.set_page_config(layout='wide')


@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


def render_app():

    st.title("🌿🌵Valley Water Rebate Plants View💐🌾")

    if 'wishlist' not in st.session_state:
        st.session_state['wishlist'] = []

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

    # Sidebar Filter
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

    ### Sidebar Wish List
    st.sidebar.divider()

    def clear_wishlist():
        st.session_state["wishlist"] = []

    def add_to_wishlist(plant_index):
        if plant_index in st.session_state.wishlist:
            st.session_state["wishlist"].remove(plant_index)
        else:
            st.session_state["wishlist"].append(plant_index)

    def remove_from_wishlist(plant_index):
        if plant_index in st.session_state.wishlist:
            st.session_state["wishlist"].remove(plant_index)

    def search_google(keyword):
        webbrowser.open_new_tab(f"https://www.google.com/search?q={keyword}")

    st.subheader(f"{len(df)} Plants Available")

    if not df.empty:
        controls = st.columns(3)
        with controls[0]:
            batch_size = st.select_slider("Batch size:",range(10,110,10))
        with controls[1]:
            row_size = st.select_slider("Row size:", range(1,6), value = 4)
        num_batches = ceil(len(df)/batch_size)
        with controls[2]:
            page = st.selectbox("Page", range(1,num_batches+1))

        batch = df[(page-1)*batch_size : page*batch_size]
        st.dataframe(batch, use_container_width=True)
        
        grid = st.columns(row_size)
        col = 0

        for idx, plant in batch.to_dict(orient='index').items():

            with grid[col]:
                button_cols = st.columns([0.8, 0.2])

                button_cols[0].button(
                    f"{plant['Scientific_Name']} | {plant['Plant_Name']} | {plant['Coverage']} sqft", 
                    key=f"google_{idx}",
                    on_click=search_google, 
                    args=[f"{plant['Scientific_Name']}"], 
                    type="primary", 
                    use_container_width=True
                )

                button_cols[1].button(
                    "🛒",
                    key=f"cart_{idx}",
                    on_click=add_to_wishlist,
                    args=[idx],
                    type="secondary",
                    use_container_width=True
                )

                for i, pic in plant['source'].items():
                    try:
                        st.image(plant['source'].get(i), use_column_width=True, caption=plant['source'].get(i))
                    except Exception as err:
                        st.write(f"☹️ error loading image due to {err}")

                st.divider()

            col = (col + 1) % row_size

    ## Show wish list
    st.sidebar.header("My Wish List")
    st.sidebar.button("Clear Wish List", use_container_width=True, on_click=clear_wishlist)
    st.sidebar.download_button(
        "Download Wish List",
        convert_df(st.session_state.orig_df.iloc[st.session_state["wishlist"]]),
        "wishlist.csv",
        "text/csv",
        key='download-csv',
        use_container_width=True
    )
    wishlist_summary = st.sidebar.container()
    if len(st.session_state["wishlist"]) > 0:
        selected = st.session_state.orig_df.iloc[st.session_state["wishlist"]]
        total_coverage = 0
        for idx, plant in selected.to_dict(orient='index').items():
            wishlist_cols = st.sidebar.columns([50, 30, 20])
            wishlist_cols[0].write(f"{plant['Scientific_Name']} | {plant['Plant_Name']}")
            qty = wishlist_cols[1].number_input("Quantity", key=f"wishlist_qty_{idx}", min_value=1, value=1, label_visibility="collapsed")
            wishlist_cols[2].button("❌", key=f"wishlist_remove_{idx}", on_click=remove_from_wishlist, args=[idx])
            try:
                st.sidebar.image(plant['source'].get("0"), use_column_width=True, caption=plant['source'].get("0"))
            except Exception as err:
                st.sidebar.write(f"☹️ error loading image due to {err}")

            total_coverage += qty * int(plant["Coverage"])

        wishlist_summary.info(f"Selected {len(selected)} Plants. Covering {total_coverage} sqft")

render_app()