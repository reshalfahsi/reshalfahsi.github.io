---
layout: page
title: Notes
permalink: /notes/
---

This page contains all of my published notes

{% for node in site.posts %}
  {% if node.title != null %}
    {% if node.layout == "post" %}
  * [{{ node.title }}]({{ node.url }})
    {% endif %}
  {% endif %}
{% endfor %}

