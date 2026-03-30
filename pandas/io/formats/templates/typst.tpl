#table(
  columns: {{ head[0] | selectattr("is_visible") | list | length }},
{% for r in head %}
  {% set visible = r | selectattr("is_visible") | list %}{% for c in visible %}[{{ c["display_value"] }}],{% if not loop.last %} {% endif%}{% endfor %}

{% endfor %}

{% for r in body %}
  {% set visible = r | selectattr("is_visible") | list %}{% for c in visible %}[{{ c["display_value"] }}],{% if not loop.last %} {% endif%}{% endfor %}

{% endfor %}
)
