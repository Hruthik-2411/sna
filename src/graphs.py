"""
Step 2 - Network construction.

For every thread and every time-window we build two graphs from the tweets that were
visible by that time (t_min <= window):

1. Cascade tree  (tweet level, directed)
       node  = tweet, edge = parent tweet -> reply
       -> captures the *shape* of the conversation (depth, breadth, virality).

2. User interaction graph  (user level, directed)
       node  = user, edge = replier -> author of the tweet replied to
       -> captures *who talks to whom* (centrality, communities, reciprocity).
"""
from __future__ import annotations

import networkx as nx
import pandas as pd


def truncate(thread_tweets: pd.DataFrame, window) -> pd.DataFrame:
    """Return the tweets an observer would have seen `window` minutes after posting."""
    if window == "full":
        return thread_tweets
    return thread_tweets[thread_tweets["t_min"] <= float(window)]


def build_cascade_tree(tw: pd.DataFrame) -> tuple[nx.DiGraph, str]:
    """Directed reply tree rooted at the source tweet.
    Replies whose parent is not (yet) visible are attached to the root, as a real-time
    observer would only know they belong to this conversation."""
    g = nx.DiGraph()
    src = tw.loc[tw["is_source"], "tweet_id"].iloc[0]
    ids = set(tw["tweet_id"])
    for r in tw.itertuples(index=False):
        g.add_node(r.tweet_id, user=r.user_id, t=r.t_min)
    for r in tw.itertuples(index=False):
        if r.is_source:
            continue
        parent = r.reply_to if (r.reply_to in ids and r.reply_to != r.tweet_id) else src
        g.add_edge(parent, r.tweet_id)
    # guard against malformed cycles: keep a BFS tree from the source
    if not nx.is_tree(g.to_undirected()) and g.number_of_nodes() > 1:
        bfs = nx.bfs_tree(g, src)
        orphans = set(g) - set(bfs)
        for o in orphans:
            bfs.add_edge(src, o)
        for n in bfs:
            bfs.nodes[n].update(g.nodes[n])
        g = bfs
    return g, src


def build_user_graph(tw: pd.DataFrame, tree: nx.DiGraph) -> nx.DiGraph:
    """Directed user graph: replier -> author of the parent tweet (self-replies dropped)."""
    ug = nx.DiGraph()
    ug.add_nodes_from(tw["user_id"].unique())
    for parent, child in tree.edges():
        u_child, u_parent = tree.nodes[child]["user"], tree.nodes[parent]["user"]
        if u_child == u_parent:
            continue
        if ug.has_edge(u_child, u_parent):
            ug[u_child][u_parent]["weight"] += 1
        else:
            ug.add_edge(u_child, u_parent, weight=1)
    return ug


def node_depths(tree: nx.DiGraph, src: str) -> dict:
    return nx.single_source_shortest_path_length(tree, src)
