---
title: "블로그"
layout: archive
permalink: categories/blog
author_profile: true
sidebar:
    nav: true
---
{% assign posts = site.categories.BLOG %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}