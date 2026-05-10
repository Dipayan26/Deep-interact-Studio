import streamlit as st
import streamlit.components.v1 as components


def request_scroll_to_top(flag_key: str) -> None:
    st.session_state[flag_key] = True


def scroll_to_top_once(flag_key: str) -> None:
    if not st.session_state.pop(flag_key, False):
        return

    components.html(
        """
        <script>
        const scrollToTop = () => {
            const parentDoc = window.parent.document;
            const appView = parentDoc.querySelector('[data-testid="stAppViewContainer"]');
            const main = parentDoc.querySelector('section.main');

            window.parent.scrollTo(0, 0);
            parentDoc.documentElement.scrollTop = 0;
            parentDoc.body.scrollTop = 0;

            if (appView && appView.scrollTo) {
                appView.scrollTo({ top: 0, left: 0, behavior: "auto" });
            }
            if (main && main.scrollTo) {
                main.scrollTo({ top: 0, left: 0, behavior: "auto" });
            }
        };

        scrollToTop();
        setTimeout(scrollToTop, 80);
        setTimeout(scrollToTop, 250);
        </script>
        """,
        height=0,
        width=0,
    )
