Assignment 1 : 

```
Traceback (most recent call last):
  File "C:\Users\Sawit\Desktop\Intern_MLE\02_train_model.py", line 78, in <module>
    train()
  File "C:\Users\Sawit\Desktop\Intern_MLE\02_train_model.py", line 40, in train
    train_df = train_df.merge(user_features, on=["user_id"]).merge(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Sawit\AppData\Roaming\Python\Python312\site-packages\pandas\core\frame.py", line 10832, in merge
    return merge(
           ^^^^^^
  File "C:\Users\Sawit\AppData\Roaming\Python\Python312\site-packages\pandas\core\reshape\merge.py", line 184, in merge
    return op.get_result(copy=copy)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Sawit\AppData\Roaming\Python\Python312\site-packages\pandas\core\reshape\merge.py", line 888, in get_result
    result = self._reindex_and_concat(
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Sawit\AppData\Roaming\Python\Python312\site-packages\pandas\core\reshape\merge.py", line 879, in _reindex_and_concat
    result = concat([left, right], axis=1, copy=copy)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Sawit\AppData\Roaming\Python\Python312\site-packages\pandas\core\reshape\concat.py", line 395, in concat
    return op.get_result()
           ^^^^^^^^^^^^^^^
  File "C:\Users\Sawit\AppData\Roaming\Python\Python312\site-packages\pandas\core\reshape\concat.py", line 684, in get_result
    new_data = concatenate_managers(
               ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Sawit\AppData\Roaming\Python\Python312\site-packages\pandas\core\internals\concat.py", line 131, in concatenate_managers
    mgrs = _maybe_reindex_columns_na_proxy(axes, mgrs_indexers, needs_copy)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Sawit\AppData\Roaming\Python\Python312\site-packages\pandas\core\internals\concat.py", line 230, in _maybe_reindex_columns_na_proxy
    mgr = mgr.copy()
          ^^^^^^^^^^
  File "C:\Users\Sawit\AppData\Roaming\Python\Python312\site-packages\pandas\core\internals\managers.py", line 593, in copy
    res = self.apply("copy", deep=deep)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Sawit\AppData\Roaming\Python\Python312\site-packages\pandas\core\internals\managers.py", line 363, in apply
    applied = getattr(b, f)(**kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Sawit\AppData\Roaming\Python\Python312\site-packages\pandas\core\internals\blocks.py", line 796, in copy
    values = values.copy()
             ^^^^^^^^^^^^^
numpy.core._exceptions._ArrayMemoryError: Unable to allocate 4.47 GiB for an array with shape (12, 50000000) and data type float64
```

