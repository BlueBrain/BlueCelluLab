{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
      :nosignatures:
   {% for item in functions %}
      ~{{ item }}
   {%- endfor %}
   {% endif %}

   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
      :nosignatures:
   {% for item in classes %}
      ~{{ item }}
   {%- endfor %}
   {% endif %}

   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
   {% for item in exceptions %}
      ~{{ item }}
   {%- endfor %}
   {% endif %}
