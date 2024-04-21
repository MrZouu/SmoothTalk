import streamlit as st
import cv2
from recognition_model_streamlit import translate_sign_language

def main():

    # Titre de la page
    st.title("🧏 SmoothTalk 🫱")
    st.header("- Présentation du Projet -")

    # Description du projet
    st.write(
        "SmoothTalk est un projet passionnant qui vise à fournir une plateforme de communication fluide et efficace. "
        "\nCe projet vise à aider les sourds et muets avec une solution innovante afin de favoriser une communication plus fluide et inclusive pour tous."
    )

    # Section fonctionnalités
    st.header("Fonctionnalités clés")
    st.markdown(
        "- **Interface Intuitive :** Une interface utilisateur conviviale pour une utilisation facile."
        "\n- **Communication en Temps Réel :** Des fonctionnalités de communication en temps réel pour une collaboration efficace."
        "\n- **Personnalisation :** Possibilité de personnaliser l'interface en fonction des préférences de l'utilisateur."
    )


    # Section équipe
    st.header("Équipe :")
    # Structure de la mise en page avec quatre colonnes
    col1, col2, col3, col4, col5 = st.columns(5)
    # Membre 1
    with col1:
        st.image("img/SmoothTalk.png", caption="Clément Auray -", use_column_width=True)

    # Membre 2
    with col2:
        st.image("img/SmoothTalk.png", caption="Lorenzo Marrocchi -", use_column_width=True)

    # Membre 3
    with col3:
        st.image("img/SmoothTalk.png", caption="Mathéo Platret -", use_column_width=True)

    # Membre 4
    with col4:
        st.image("img/SmoothTalk.png", caption="Evann Ali-Yahia -", use_column_width=True)

    # Membre 5
    with col5:
        st.image("img/SmoothTalk.png", caption="Thomas Ramade -", use_column_width=True)

    # Pied de page
    st.markdown(
        "Pour plus d'informations, consultez [notre site web](http://www.votresite.com)"
    )


    st.header("Détection en temps réel : langue des signes.")

    # Création d'une zone vidéo pour afficher le flux
    video_placeholder = st.empty()

    # Fonction pour afficher la vidéo de la webcam et les images détectées
    def show_video():
        # Capture vidéo à partir de la webcam
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break

            # Appel de la fonction pour détecter et traduire la langue des signes
            translated_img = translate_sign_language(img)

            # Affichage de la vidéo de la webcam dans la zone vidéo
            video_placeholder.image(translated_img, channels="BGR", use_column_width=True)

    # Appel de la fonction pour afficher la vidéo de la webcam
    show_video()




if __name__ == "__main__":
    main()
