/* One store: state, dispatch, subscribe. No DOM, importable under node. */

export function createStore(reducer, initial, { onDispatch } = {}) {
  let state = initial;
  const listeners = new Set();
  let dispatching = false;

  function getState() {
    return state;
  }

  function dispatch(action) {
    if (!action || typeof action.type !== "string") return state;
    const prev = state;
    const next = reducer(prev, action);
    if (next === prev) return state;
    state = next;
    if (onDispatch) onDispatch(action, prev, next);
    if (dispatching) return state;
    dispatching = true;
    try {
      for (const listener of [...listeners]) listener(state, prev, action);
    } finally {
      dispatching = false;
    }
    return state;
  }

  function subscribe(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  return { getState, dispatch, subscribe };
}
