from pyecharts import options as opts
from pyecharts.charts import Bar

c = (
    Bar()
    .add_dataset(
        source=[
            ["product", "amount"],
            ["swning-tricycle", 3243],
            ["tricycle", 4803],
            ["bus", 5926],
            ["bicycle", 10477],
            ["truck", 12871],
            ["van", 24950],
            ["people", 27059],
            ["motor", 29642],
            ["pedestrian", 79337],
            ["car", 144865]
        ]
    )
    .add_yaxis(
        series_name="",
        y_axis=[],
        # yaxis_data=[],
        encode={"y": "product", "x": "amount"},
        label_opts=opts.LabelOpts(is_show=False),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Dataset normal bar example"),
        xaxis_opts=opts.AxisOpts(name="amount"),
        yaxis_opts=opts.AxisOpts(type_="category"),
        visualmap_opts=opts.VisualMapOpts(
            orient="vertical",
            pos_left="center",
            min_=10,
            max_=100,
            # range_text=["High Score", "Low Score"],
            dimension=0,
            range_color=["#D7DA8B", "#E15457"],
        ),
    )
    .render("dataset_bar_1.html")
)
